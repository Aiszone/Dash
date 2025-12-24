from __future__ import annotations

import pandas as pd
import yfinance as yf


def _force_series(obj, name: str = "Close") -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if "Close" in obj.columns:
            s = obj["Close"]
            if isinstance(s, pd.DataFrame):
                if s.shape[1] >= 1:
                    s = s.iloc[:, 0]
                else:
                    s = pd.Series(dtype=float)
        else:
            s = obj.squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0] if s.shape[1] >= 1 else pd.Series(dtype=float)
    else:
        s = pd.Series(obj)

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")

    s = s.loc[~s.index.isna()].sort_index()
    s = s[~s.duplicated(keep="last")]

    s = pd.to_numeric(s, errors="coerce").astype(float).dropna()
    s.name = s.name or name
    return s


def fetch_series(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.Series(dtype=float, name=ticker)

    s = _force_series(df, name="Close")
    s.name = ticker
    return s


def to_monthly_bm(s: pd.Series | pd.DataFrame) -> pd.Series:
    s1 = _force_series(s)
    sm = s1.resample("BM").last().dropna()
    sm.name = s1.name
    return sm