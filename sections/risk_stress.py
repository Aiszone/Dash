import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from lib.data import fetch_series, to_monthly_bm
from lib.ui import panel, section_header


def _compute_portfolio_returns(prices: pd.DataFrame, w: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    if prices is None or prices.empty:
        raise ValueError("No price data available to compute returns.")

    r_assets = prices.pct_change().dropna(how="all")
    if r_assets.empty:
        raise ValueError("Unable to compute monthly returns (empty series).")

    r_assets = r_assets.dropna(axis=1, how="all")
    if r_assets.empty:
        raise ValueError("All return columns are empty.")

    w = np.asarray(w, dtype=float).reshape(-1)
    if len(w) != r_assets.shape[1]:
        raise ValueError("Weight vector length does not match the number of assets.")

    r_port = pd.Series(r_assets.values @ w, index=r_assets.index, name="portfolio_returns").dropna()
    if len(r_port) < 3:
        raise ValueError("Portfolio return series is too short for risk statistics.")

    return r_assets, r_port


def _compute_var_cvar_historical(r_port: pd.Series, confidence: float) -> tuple[float, float]:
    q = r_port.quantile(1 - confidence)
    tail = r_port[r_port <= q]
    if tail.empty:
        v = float(-q)
        return v, v
    return float(-q), float(-tail.mean())


def _cholesky_psd(sigma: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(sigma)
        vals = np.clip(vals, 1e-12, None)
        sigma_psd = vecs @ np.diag(vals) @ vecs.T
        return np.linalg.cholesky(sigma_psd)


def _compute_var_cvar_mc(r_assets: pd.DataFrame, w: np.ndarray, confidence: float, n_sim: int) -> tuple[float, float]:
    if len(r_assets) < 3:
        raise ValueError("Not enough observations to run Monte Carlo VaR.")

    mu = r_assets.mean().values
    sigma = r_assets.cov().values
    n = len(mu)

    l = _cholesky_psd(sigma)
    z = np.random.normal(size=(n_sim, n))
    r_sim = mu + z @ l.T

    w = np.asarray(w, dtype=float).reshape(-1)
    r_port_sim = r_sim @ w

    q = np.quantile(r_port_sim, 1 - confidence)
    tail = r_port_sim[r_port_sim <= q]
    if tail.size == 0:
        v = float(-q)
        return v, v

    return float(-q), float(-tail.mean())


def _compute_volatility_contribution(
    r_assets: pd.DataFrame, w: np.ndarray, asset_labels: list[str]
) -> tuple[float, pd.DataFrame]:
    sigma = r_assets.cov().values
    w = np.asarray(w, dtype=float).reshape(-1)

    sigma_p = float(np.sqrt(w @ sigma @ w))
    if sigma_p <= 0:
        raise ValueError("Portfolio volatility is zero or undefined.")

    mrc = sigma @ w / sigma_p
    rc_abs = w * mrc
    rc_pct = rc_abs / sigma_p

    df = (
        pd.DataFrame(
            {
                "Asset": asset_labels,
                "Weight (%)": w * 100.0,
                "Vol contrib (%)": rc_pct * 100.0,
            }
        )
        .sort_values("Vol contrib (%)", ascending=False)
        .reset_index(drop=True)
    )
    return sigma_p, df


HISTORICAL_SCENARIOS = {
    "Dot-com bust (2000–2002)": ("2000-03-01", "2002-09-30"),
    "Subprime crisis (2007–2009)": ("2007-07-01", "2009-03-31"),
    "GFC core (Sep 2008–Mar 2009)": ("2008-09-01", "2009-03-31"),
    "Flash crash (2010)": ("2010-04-01", "2010-09-30"),
    "Eurozone crisis (2011–2012)": ("2011-07-01", "2012-06-30"),
    "Taper tantrum (2013)": ("2013-05-01", "2013-12-31"),
    "China equity crash (2015)": ("2015-06-01", "2016-02-29"),
    "COVID shock (2020)": ("2020-02-01", "2020-04-30"),
    "Inflation regime (2022)": ("2022-01-01", "2022-12-31"),
}


def _fetch_crisis_prices_for_selection(
    universe: dict,
    selection_labels: list[str],
    w: np.ndarray,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    series_dict: dict[str, pd.Series] = {}
    weights_dict: dict[str, float] = {}

    w = np.asarray(w, dtype=float).reshape(-1)
    if len(selection_labels) != len(w):
        raise ValueError("Selection labels and weights do not match.")

    for label, weight in zip(selection_labels, w):
        if label not in universe:
            continue
        ticker, _ = universe[label]
        s = fetch_series(ticker, start, end)
        sm = to_monthly_bm(s)
        if sm is None or sm.empty:
            continue
        series_dict[label] = sm
        weights_dict[label] = float(weight)

    if not series_dict:
        return pd.DataFrame(), np.array([])

    prices = pd.concat(series_dict.values(), axis=1)
    prices.columns = list(series_dict.keys())
    prices = prices.sort_index().dropna(axis=1, how="all")
    if prices.empty:
        return pd.DataFrame(), np.array([])

    labels_used = list(prices.columns)
    w_used = np.array([weights_dict[l] for l in labels_used], dtype=float)
    total_w = float(w_used.sum())
    if total_w <= 0:
        return pd.DataFrame(), np.array([])

    w_used = w_used / total_w
    return prices, w_used


def _compute_crisis_metrics(prices_crisis: pd.DataFrame, w_crisis: np.ndarray) -> dict:
    if prices_crisis.empty or w_crisis.size == 0:
        return {"cum_return": np.nan, "max_dd": np.nan, "worst_month": np.nan, "n_months": 0}

    r = prices_crisis.pct_change().dropna(how="all").dropna(axis=1, how="all")
    if r.empty:
        return {"cum_return": np.nan, "max_dd": np.nan, "worst_month": np.nan, "n_months": 0}

    if len(w_crisis) != r.shape[1]:
        raise ValueError("Crisis return matrix and weights are inconsistent.")

    r_port = pd.Series(r.values @ w_crisis, index=r.index, name="crisis_portfolio_returns").dropna()
    if r_port.empty:
        return {"cum_return": np.nan, "max_dd": np.nan, "worst_month": np.nan, "n_months": 0}

    cum_return = float((1 + r_port).prod() - 1.0)
    nav = (1 + r_port).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    max_dd = float(dd.min())
    worst_month = float(r_port.min())

    return {"cum_return": cum_return, "max_dd": max_dd, "worst_month": worst_month, "n_months": int(len(r_port))}


FACTOR_DEFS = {
    "US stocks": {"group": "Equity"},
    "Europe stocks": {"group": "Equity"},
    "China stocks": {"group": "Equity"},
    "Emerging stocks": {"group": "Equity"},
    "US small caps": {"group": "Equity"},
    "Tech": {"group": "Equity"},
    "US bonds": {"group": "Rates"},
    "Global bonds": {"group": "Rates"},
    "High yield": {"group": "Credit"},
    "Gold": {"group": "Commodities"},
    "Silver": {"group": "Commodities"},
    "Oil": {"group": "Commodities"},
    "BTC": {"group": "Crypto"},
    "Other": {"group": "Other"},
}

CATEGORY_BY_TICKER = {
    "SPY": "US stocks",
    "QQQ": "Tech",
    "VT": "US stocks",
    "VGK": "Europe stocks",
    "EEM": "Emerging stocks",
    "INDA": "Emerging stocks",
    "MCHI": "China stocks",
    "TLT": "US bonds",
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil",
    "BTC-USD": "BTC",
    "ETH-USD": "BTC",
}

FACTOR_TICKERS = {
    "US stocks": "SPY",
    "Europe stocks": "VGK",
    "China stocks": "MCHI",
    "Emerging stocks": "EEM",
    "US small caps": "IWM",
    "Tech": "QQQ",
    "US bonds": "TLT",
    "Global bonds": "BNDX",
    "High yield": "HYG",
    "Gold": "GLD",
    "Silver": "SLV",
    "Oil": "USO",
    "BTC": "BTC-USD",
}

FACTOR_CORR = {
    "US stocks": {
        "Tech": 0.85,
        "Europe stocks": 0.75,
        "Emerging stocks": 0.65,
        "US small caps": 0.80,
        "US bonds": -0.20,
        "Global bonds": -0.15,
        "High yield": 0.60,
        "Gold": -0.15,
        "Silver": -0.10,
        "Oil": 0.30,
        "BTC": 0.25,
    },
    "Tech": {
        "Europe stocks": 0.70,
        "Emerging stocks": 0.60,
        "US small caps": 0.70,
        "US bonds": -0.25,
        "Gold": -0.20,
        "BTC": 0.30,
    },
    "Europe stocks": {
        "Emerging stocks": 0.60,
        "US bonds": -0.10,
        "Global bonds": -0.05,
        "Gold": -0.10,
        "Oil": 0.25,
    },
    "Emerging stocks": {
        "US bonds": -0.25,
        "Global bonds": -0.20,
        "Gold": -0.25,
        "Oil": 0.35,
        "BTC": 0.30,
    },
    "US bonds": {"Global bonds": 0.80, "High yield": 0.40, "Gold": 0.10, "BTC": -0.10},
    "Global bonds": {"High yield": 0.35, "Gold": 0.10},
    "High yield": {"Emerging stocks": 0.55, "BTC": 0.20},
    "Gold": {"Silver": 0.75, "Oil": 0.10, "BTC": -0.05},
    "Silver": {"Oil": 0.20, "BTC": 0.05},
    "Oil": {"BTC": 0.15},
    "BTC": {},
}


def _build_hypothetical_categories(universe: dict, selection_labels: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for label in selection_labels:
        if label not in universe:
            continue
        ticker, _ = universe[label]
        out[label] = CATEGORY_BY_TICKER.get(ticker, "Other")
    return out


def _get_factor_corr_macro(f_target: str, f_source: str) -> float:
    if f_target == f_source:
        return 1.0
    d1 = FACTOR_CORR.get(f_target, {})
    if f_source in d1:
        return float(d1[f_source])
    d2 = FACTOR_CORR.get(f_source, {})
    if f_target in d2:
        return float(d2[f_target])
    return 0.0


@st.cache_data(show_spinner=False)
def _compute_dynamic_factor_corr(factors: tuple[str, ...], horizon_days: int = 365) -> pd.DataFrame | None:
    factors = tuple(sorted(set(factors)))
    if not factors:
        return None

    end = dt.date.today()
    start = end - dt.timedelta(days=int(horizon_days))

    series_dict: dict[str, pd.Series] = {}
    for f in factors:
        ticker = FACTOR_TICKERS.get(f)
        if not ticker:
            continue
        try:
            s = fetch_series(ticker, start.isoformat(), end.isoformat())
        except Exception:
            continue
        if s is None or s.empty:
            continue
        series_dict[f] = s

    if not series_dict:
        return None

    prices = pd.concat(series_dict.values(), axis=1)
    prices.columns = list(series_dict.keys())
    prices = prices.sort_index().dropna(how="all")
    if prices.shape[1] < 2:
        return None

    r = prices.pct_change().dropna(how="any")
    if len(r) < 60:
        return None

    return r.corr().clip(-0.95, 0.95)


def _get_factor_corr_mixed(
    f_target: str,
    f_source: str,
    corr_dyn: pd.DataFrame | None,
    w_macro: float,
    w_dyn: float,
) -> float:
    corr_macro = _get_factor_corr_macro(f_target, f_source)

    if corr_dyn is None:
        return corr_macro
    if (f_target not in corr_dyn.index) or (f_source not in corr_dyn.columns):
        return corr_macro

    corr_d = float(corr_dyn.loc[f_target, f_source])
    if np.isnan(corr_d):
        return corr_macro

    diff = corr_d - corr_macro
    corr_d_adj = corr_macro + float(np.clip(diff, -0.4, 0.4))

    if "BTC" in (f_target, f_source):
        w_dyn_eff = min(1.0, w_dyn * 1.5)
        w_macro_eff = 1.0 - w_dyn_eff
    else:
        w_dyn_eff = w_dyn
        w_macro_eff = w_macro

    corr_final = w_macro_eff * corr_macro + w_dyn_eff * corr_d_adj
    return float(np.clip(corr_final, -0.95, 0.95))


def _build_factor_corr_df(
    factors: list[str],
    corr_dyn: pd.DataFrame | None,
    w_macro: float,
    w_dyn: float,
) -> pd.DataFrame:
    data: dict[str, dict[str, float]] = {}
    for f_i in factors:
        row: dict[str, float] = {}
        for f_j in factors:
            row[f_j] = _get_factor_corr_mixed(f_i, f_j, corr_dyn, w_macro, w_dyn)
        data[f_i] = row
    return pd.DataFrame(data, index=factors)


def render_risk_stress(
    prices: pd.DataFrame,
    w: np.ndarray,
    nav10k: pd.Series,
    universe: dict,
    selection: list[tuple[str, float]],
) -> None:
    section_header("Risk & Stress", "VaR/CVaR, volatility contribution, historical crises, and factor shocks.")

    try:
        r_assets, r_port = _compute_portfolio_returns(prices, w)
    except Exception as e:
        st.error(f"Unable to compute returns: {e}")
        return

    with panel("Loss risk (VaR / CVaR)", "Monthly horizon. Historical + Monte Carlo."):
        c1, c2 = st.columns([1.1, 1.6])
        with c1:
            conf_label = st.selectbox("Confidence", ["95%", "99%"], index=0, key="risk_confidence")
            confidence = 0.95 if "95" in conf_label else 0.99
        with c2:
            n_sim = st.slider("Monte Carlo simulations", 1_000, 50_000, 10_000, 1_000, key="risk_n_sim")

        conf_pct = int(confidence * 100)
        var_hist, cvar_hist = _compute_var_cvar_historical(r_port, confidence)
        try:
            var_mc, cvar_mc = _compute_var_cvar_mc(r_assets, w, confidence, int(n_sim))
        except Exception:
            var_mc, cvar_mc = np.nan, np.nan

        top = st.columns(4)
        top[0].dataframe(pd.DataFrame([{"Hist VaR": f"{var_hist*100:.2f}%".replace(".", ",")}]), hide_index=True, use_container_width=True)
        top[1].dataframe(pd.DataFrame([{"Hist CVaR": f"{cvar_hist*100:.2f}%".replace(".", ",")}]), hide_index=True, use_container_width=True)
        top[2].dataframe(pd.DataFrame([{"MC VaR": ("—" if np.isnan(var_mc) else f"{var_mc*100:.2f}%".replace(".", ","))}]), hide_index=True, use_container_width=True)
        top[3].dataframe(pd.DataFrame([{"MC CVaR": ("—" if np.isnan(cvar_mc) else f"{cvar_mc*100:.2f}%".replace(".", ","))}]), hide_index=True, use_container_width=True)

        bottom = st.columns(2)
        bottom[0].dataframe(
            pd.DataFrame([{"Impact (Hist VaR) on $10,000": f"${var_hist*10_000:,.0f}".replace(",", " ")}]),
            hide_index=True,
            use_container_width=True,
        )
        bottom[1].dataframe(
            pd.DataFrame([{"Impact (MC VaR) on $10,000": ("—" if np.isnan(var_mc) else f"${var_mc*10_000:,.0f}".replace(",", " "))}]),
            hide_index=True,
            use_container_width=True,
        )

    with panel("Volatility contribution", "Who drives portfolio risk (monthly vol)."):
        try:
            asset_labels = list(prices.columns)
            sigma_p, df_vol = _compute_volatility_contribution(r_assets, w, asset_labels)
            top_asset = str(df_vol.iloc[0]["Asset"])
            top_pct = float(df_vol.iloc[0]["Vol contrib (%)"])

            c1, c2 = st.columns(2)
            c1.dataframe(
                pd.DataFrame([{"Total volatility (monthly)": f"{sigma_p*100:.2f}%".replace(".", ",")}]),
                hide_index=True,
                use_container_width=True,
            )
            c2.dataframe(
                pd.DataFrame([{"Top contributor": f"{top_asset} — {top_pct:.1f}%".replace(".", ",")}]),
                hide_index=True,
                use_container_width=True,
            )

            df_disp = df_vol.copy()
            df_disp["Weight (%)"] = df_disp["Weight (%)"].round(1)
            df_disp["Vol contrib (%)"] = df_disp["Vol contrib (%)"].round(1)
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Unable to compute volatility contribution: {e}")

    with panel("Historical stress tests", "Replay crisis windows on the current portfolio."):
        available = list(HISTORICAL_SCENARIOS.keys())
        selected = st.multiselect(
            "Crises to test",
            options=available,
            default=available,
            key="hist_stress_select",
        )

        selection_labels = [lbl for (lbl, _w) in selection]
        w_vec = np.asarray(w, dtype=float).reshape(-1)

        rows_hist: list[dict] = []
        for scen_name in selected:
            start, end = HISTORICAL_SCENARIOS[scen_name]
            try:
                prices_crisis, w_crisis = _fetch_crisis_prices_for_selection(universe, selection_labels, w_vec, start, end)
                metrics = _compute_crisis_metrics(prices_crisis, w_crisis)
            except Exception:
                metrics = {"cum_return": np.nan, "max_dd": np.nan, "worst_month": np.nan, "n_months": 0}

            rows_hist.append(
                {
                    "Scenario": scen_name,
                    "Window": f"{start} → {end}",
                    "Cumulative return (%)": (metrics["cum_return"] * 100.0) if np.isfinite(metrics["cum_return"]) else np.nan,
                    "Max drawdown (%)": (metrics["max_dd"] * 100.0) if np.isfinite(metrics["max_dd"]) else np.nan,
                    "Worst month (%)": (metrics["worst_month"] * 100.0) if np.isfinite(metrics["worst_month"]) else np.nan,
                    "Months": metrics["n_months"],
                    "$10,000 PnL": (metrics["cum_return"] * 10_000.0) if np.isfinite(metrics["cum_return"]) else np.nan,
                }
            )

        if rows_hist:
            df_hist = pd.DataFrame(rows_hist)
            st.dataframe(
                df_hist.style.format(
                    {
                        "Cumulative return (%)": "{:.2f}",
                        "Max drawdown (%)": "{:.2f}",
                        "Worst month (%)": "{:.2f}",
                        "$10,000 PnL": "${:,.0f}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info("Select at least one scenario to view results.")

    with panel("Hypothetical stress tests", "Create your own crisis with factor shocks (propagated via correlations)."):
        selection_labels = [lbl for (lbl, _w) in selection]
        categories_by_label = _build_hypothetical_categories(universe, selection_labels)
        factors_in_portfolio = sorted(set(categories_by_label.values()) - {"Other"})
        if not factors_in_portfolio:
            st.warning("Unable to map the portfolio assets to factor categories.")
            return

        all_factors = [f for f in FACTOR_DEFS.keys() if f != "Other"]
        default_factors = [f for f in factors_in_portfolio if f in all_factors] or all_factors

        factors_selected = st.multiselect(
            "Factors receiving a DIRECT shock (%)",
            options=all_factors,
            default=default_factors,
            key="hypo_factors_select",
        )
        if not factors_selected:
            st.info("Select at least one factor.")
            return

        shocks_direct: dict[str, float] = {}
        cols = st.columns(min(3, len(factors_selected)))
        for i, factor in enumerate(factors_selected):
            group = FACTOR_DEFS.get(factor, {"group": "Other"}).get("group", "Other")
            if group in ("Rates", "Credit"):
                min_val, max_val = -50.0, 50.0
            elif group == "Crypto":
                min_val, max_val = -100.0, 200.0
            elif group == "Commodities":
                min_val, max_val = -80.0, 80.0
            else:
                min_val, max_val = -80.0, 80.0

            with cols[i % len(cols)]:
                shocks_direct[factor] = (
                    st.slider(
                        f"{factor}",
                        min_value=min_val,
                        max_value=max_val,
                        value=0.0,
                        step=1.0,
                        key=f"shock_direct_{factor}",
                    )
                    / 100.0
                )

        relevant_factors = sorted(set(factors_in_portfolio) | set(factors_selected))

        with st.expander("Correlation assumptions"):
            horizon_label = st.radio(
                "Dynamic correlation lookback",
                ["6 months", "1 year", "2 years"],
                index=1,
                horizontal=True,
                key="corr_horizon",
            )
            horizon_days = {"6 months": 180, "1 year": 365, "2 years": 730}[horizon_label]

            w_dyn = st.slider(
                "Weight on recent market regime",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="corr_dyn_weight",
            )
            w_macro = 1.0 - w_dyn

            corr_dyn = _compute_dynamic_factor_corr(tuple(relevant_factors), horizon_days=horizon_days)
            corr_used = _build_factor_corr_df(relevant_factors, corr_dyn, w_macro=w_macro, w_dyn=w_dyn)
            st.dataframe(corr_used.style.format("{:.2f}"), use_container_width=True)

        shocks_effective: dict[str, float] = {}
        for f_target in relevant_factors:
            shock_sum = 0.0
            for f_src, shock_src in shocks_direct.items():
                corr = _get_factor_corr_mixed(f_target, f_src, corr_dyn, w_macro=w_macro, w_dyn=w_dyn)
                shock_sum += corr * shock_src
            shocks_effective[f_target] = shock_sum

        rows_hypo: list[dict] = []
        total_impact = 0.0
        total_weight = 0.0

        w_vec = np.asarray(w, dtype=float).reshape(-1)
        for (label, _), w_i in zip(selection, w_vec):
            factor = categories_by_label.get(label, "Other")
            shock_factor = shocks_effective.get(factor, 0.0)
            impact = float(w_i * shock_factor)

            rows_hypo.append(
                {
                    "Asset": label,
                    "Factor": factor,
                    "Weight": float(w_i),
                    "Applied shock (%)": shock_factor * 100.0,
                    "Portfolio impact (%)": impact * 100.0,
                    "$10,000 PnL": impact * 10_000.0,
                }
            )

            total_impact += impact
            total_weight += float(w_i)

        if total_weight != 0:
            rows_hypo.append(
                {
                    "Asset": "TOTAL",
                    "Factor": "",
                    "Weight": total_weight,
                    "Applied shock (%)": np.nan,
                    "Portfolio impact (%)": total_impact * 100.0,
                    "$10,000 PnL": total_impact * 10_000.0,
                }
            )

        df_hypo = pd.DataFrame(rows_hypo)

        st.dataframe(
            df_hypo.style.format(
                {
                    "Weight": "{:.3f}",
                    "Applied shock (%)": "{:.1f}",
                    "Portfolio impact (%)": "{:.2f}",
                    "$10,000 PnL": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )