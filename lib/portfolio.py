from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import fetch_series, to_monthly_bm


BASE = 10_000.0


def _force_named_series(obj, fallback_name: str) -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        s = obj.squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
    else:
        s = pd.Series(obj)
    if s.name is None:
        s.name = fallback_name
    return s


def build_prices_usd_monthly(
    selection: list[tuple[str, float]],
    universe: dict[str, tuple[str, str]],
    start: str,
    end: str,
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}

    for label, _w in selection:
        if label not in universe:
            continue
        ticker, _ = universe[label]
        s = fetch_series(ticker, start, end)
        sm = to_monthly_bm(s)
        sm = _force_named_series(sm, label)
        sm.name = label
        if not sm.empty:
            cols[label] = sm

    if not cols:
        return pd.DataFrame()

    df = pd.concat(cols.values(), axis=1, join="inner").dropna(how="any")
    df.columns = list(cols.keys())
    return df


def weights_vector(selection: list[tuple[str, float]]) -> pd.Series:
    if not selection:
        return pd.Series(dtype=float)
    w = pd.Series({lbl: float(w) for lbl, w in selection}, dtype=float)
    total = float(w.sum())
    return (w / total) if total > 0 else w


def nav_base_10k(prices_bm: pd.DataFrame, w: pd.Series, base: float = BASE) -> pd.Series:
    if prices_bm is None or prices_bm.empty or w is None or w.empty:
        return pd.Series(dtype=float, name="NAV_10k")

    cols = [c for c in w.index if c in prices_bm.columns]
    if not cols:
        return pd.Series(dtype=float, name="NAV_10k")

    aligned = prices_bm[cols].dropna(how="any")
    if aligned.empty or len(aligned) < 2:
        return pd.Series(dtype=float, name="NAV_10k")

    wv = w.reindex(cols).astype(float)
    if wv.sum() > 0:
        wv = wv / float(wv.sum())

    rebase = aligned / aligned.iloc[0]
    nav = (rebase.mul(wv, axis=1)).sum(axis=1) * float(base)
    nav.name = "NAV_10k"
    return nav


def compute_stats_from_nav(nav10k: pd.Series) -> dict:
    if nav10k is None or len(nav10k.dropna()) < 3:
        return dict(cagr=np.nan, vol_ann=np.nan, sharpe=np.nan, nav=np.nan)

    nav10k = nav10k.dropna().astype(float)
    r = np.log(nav10k / nav10k.shift(1)).dropna()
    if r.empty:
        return dict(cagr=np.nan, vol_ann=np.nan, sharpe=np.nan, nav=float(nav10k.iloc[-1]))

    mean_m = float(r.mean())
    std_m = float(r.std(ddof=1))

    cagr = float(np.exp(mean_m * 12.0) - 1.0)
    vol_ann = float(std_m * math.sqrt(12.0))
    sharpe = float((mean_m * 12.0) / vol_ann) if vol_ann > 0 else np.nan

    return dict(cagr=cagr, vol_ann=vol_ann, sharpe=sharpe, nav=float(nav10k.iloc[-1]))


def benchmarks_base_10k(
    tickers: list[str],
    start: str,
    end: str,
    ref_index: pd.Index | None = None,
    base: float = BASE,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    cols: dict[str, pd.Series] = {}
    for tkr in tickers:
        s = fetch_series(tkr, start, end)
        sm = to_monthly_bm(s)
        sm = _force_named_series(sm, tkr)
        sm.name = tkr
        if not sm.empty:
            cols[tkr] = sm

    if not cols:
        return pd.DataFrame()

    df = pd.concat(cols.values(), axis=1, join="inner").dropna(how="any")
    df.columns = list(cols.keys())

    if ref_index is not None and len(ref_index) > 0:
        df = df.reindex(ref_index).dropna(how="any")

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    return df.div(df.iloc[0], axis=1) * float(base)


def relative_stats(port_nav10k: pd.Series, bench_nav10k: pd.Series) -> dict:
    P = _force_named_series(port_nav10k, "P")
    B = _force_named_series(bench_nav10k, "B")
    P.name = "P"
    B.name = "B"

    df = pd.concat([P, B], axis=1, join="inner").dropna()
    if df.empty or len(df) < 2:
        return {k: np.nan for k in ["excess_cumul", "cagr_diff", "tracking_err", "ir", "win_ratio"]}

    excess_cumul = (df["P"].iloc[-1] / df["P"].iloc[0]) / (df["B"].iloc[-1] / df["B"].iloc[0]) - 1.0

    rp = np.log(df["P"] / df["P"].shift(1)).dropna()
    rb = np.log(df["B"] / df["B"].shift(1)).dropna()
    r = pd.concat([rp, rb], axis=1, join="inner").dropna()
    r.columns = ["rp", "rb"]

    if r.empty or len(r) < 2:
        return {
            "excess_cumul": float(excess_cumul),
            "cagr_diff": np.nan,
            "tracking_err": np.nan,
            "ir": np.nan,
            "win_ratio": np.nan,
        }

    diff = r["rp"] - r["rb"]
    mean_diff_m = float(diff.mean())
    std_diff_m = float(diff.std(ddof=1))

    cagr_diff = float(np.exp(mean_diff_m * 12.0) - 1.0)
    tracking_err = float(std_diff_m * math.sqrt(12.0)) if std_diff_m > 0 else np.nan
    ir = float((mean_diff_m * 12.0) / tracking_err) if (tracking_err is not None and tracking_err > 0) else np.nan
    win_ratio = float((diff > 0).mean())

    return {
        "excess_cumul": float(excess_cumul),
        "cagr_diff": cagr_diff,
        "tracking_err": tracking_err,
        "ir": ir,
        "win_ratio": win_ratio,
    }