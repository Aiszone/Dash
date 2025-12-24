from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.ui import apply_plotly_style, render_table_report, section_header


def _force_series(x, name: str = "NAV") -> pd.Series:
    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0].copy()
        s.name = s.name or name
    elif isinstance(x, pd.Series):
        s = x.copy()
        s.name = s.name or name
    else:
        raise TypeError("nav must be a pandas Series or a single-column DataFrame.")
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _monthly_returns(nav: pd.Series, start: str | None = None, end: str | None = None) -> pd.Series:
    s = nav.loc[start:end] if (start or end) else nav
    nav_m = s.resample("M").last()
    r = nav_m.pct_change().dropna()
    r.name = "Monthly return"
    return r


def _fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_money(x: float, symbol: str = "$", decimals: int = 0) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{symbol}{x:,.{decimals}f}".replace(",", " ")


def _simulate_paths(mu_m: float, sigma_m: float, months: int, n_sims: int, initial: float) -> tuple[np.ndarray, np.ndarray]:
    if months <= 0 or n_sims <= 0:
        raise ValueError("months and n_sims must be > 0.")
    ret_paths = np.random.normal(loc=mu_m, scale=sigma_m, size=(months, n_sims))
    nav_paths = np.empty((months + 1, n_sims), dtype=float)
    nav_paths[0, :] = initial
    nav_paths[1:, :] = initial * np.cumprod(1.0 + ret_paths, axis=0)
    return nav_paths, ret_paths


def _percentile_paths(nav_paths: np.ndarray, percentiles: tuple[float, ...] = (2.3, 15.9, 50.0, 84.1, 97.7)) -> pd.DataFrame:
    perc = np.percentile(nav_paths, q=list(percentiles), axis=1).T
    df = pd.DataFrame(perc, columns=[f"{p:.1f}%" for p in percentiles])
    df.index.name = "month"
    return df


def _label_from_percentile(p_label: str) -> str:
    mapping = {
        "97.7%": "Best (+2σ)",
        "84.1%": "Good (+1σ)",
        "50.0%": "Median",
        "15.9%": "Bad (−1σ)",
        "2.3%": "Worst (−2σ)",
    }
    return mapping.get(p_label, p_label)


def _build_percentile_chart(df_paths: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    months = df_paths.index.values
    years = months / 12.0
    mask = months >= 1

    fig = go.Figure()
    for col in df_paths.columns:
        fig.add_scatter(
            x=years[mask],
            y=df_paths[col].values[mask],
            mode="lines",
            line=dict(width=2),
            name=_label_from_percentile(col),
            hovertemplate=f"Year: %{{x:.0f}}<br>Value: {currency_symbol}%{{y:,.0f}}<extra></extra>",
        )

    fig.update_layout(
        title="Simulated portfolio paths from historical monthly returns",
        xaxis_title="Horizon (years)",
        yaxis_title="Portfolio value",
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
        legend=dict(orientation="v", x=1.02, y=1, yanchor="top", xanchor="left"),
    )
    apply_plotly_style(fig)
    return fig


def _summary_rows(nav_paths: np.ndarray, ret_paths: np.ndarray, horizon_years: int, initial: float) -> pd.DataFrame:
    months, _ = ret_paths.shape
    years = float(horizon_years)

    final_nav = nav_paths[-1, :]
    vol_ann_per_sim = np.std(ret_paths, axis=0, ddof=1) * np.sqrt(12.0)

    pcts = [2.3, 15.9, 50.0, 84.1, 97.7]
    cuts = np.percentile(final_nav, pcts)

    bounds = [
        (-np.inf, cuts[0]),
        (cuts[0], cuts[1]),
        (cuts[1], cuts[2]),
        (cuts[2], cuts[3]),
        (cuts[3], cuts[4]),
    ]
    labels = ["2.3%", "15.9%", "50.0%", "84.1%", "97.7%"]

    rows = []
    for (lo, hi), key in zip(bounds, labels):
        m = (final_nav <= hi) & (final_nav > lo)
        if not np.any(m):
            nav_star = float(np.percentile(final_nav, float(key.rstrip("%"))))
            vol_sim = float(np.median(vol_ann_per_sim))
        else:
            nav_star = float(np.percentile(final_nav[m], float(key.rstrip("%"))))
            vol_sim = float(np.median(vol_ann_per_sim[m]))

        cagr = (nav_star / float(initial)) ** (1.0 / years) - 1.0
        rows.append(
            {
                "Percentile band": _label_from_percentile(key),
                "Terminal value": nav_star,
                "CAGR": cagr,
                "Simulated vol (ann.)": vol_sim,
            }
        )

    return pd.DataFrame(rows)


def render_monte_carlo(nav, *, start: str | None = None, end: str | None = None) -> None:
    section_header("Monte Carlo simulation", "Percentile fan chart and scenario summary table.")

    s_nav = _force_series(nav, name="NAV")
    r_m = _monthly_returns(s_nav, start, end)

    if r_m.size < 24:
        st.warning("At least 24 monthly returns are required for a reliable Monte Carlo simulation.")
        return

    mu_m = float(r_m.mean())
    sigma_m = float(r_m.std(ddof=1))

    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0], vertical_alignment="center")
    with c1:
        initial = st.number_input(
            "Initial capital",
            min_value=1_000,
            max_value=10_000_000,
            value=int(float(s_nav.iloc[-1])),
            step=1_000,
            key="mc_initial",
        )
    with c2:
        horizon_years = st.slider("Horizon (years)", min_value=1, max_value=40, value=30, step=1, key="mc_horizon")
    with c3:
        n_sims = st.number_input("Simulations", min_value=100, max_value=5_000, value=600, step=100, key="mc_sims")
    with c4:
        st.metric("μ (monthly, hist.)", _fmt_pct(mu_m, 2))
        st.metric("σ (monthly, hist.)", _fmt_pct(sigma_m, 2))

    use_seed = st.toggle("Deterministic run (seeded)", value=True)
    seed = 42 if use_seed else None

    months = int(horizon_years * 12)
    if seed is not None:
        np.random.seed(seed)

    nav_paths, ret_paths = _simulate_paths(mu_m, sigma_m, months=months, n_sims=int(n_sims), initial=float(initial))

    df_perc = _percentile_paths(nav_paths)
    fig = _build_percentile_chart(df_perc, currency_symbol="$")
    st.plotly_chart(fig, use_container_width=True)

    export = df_perc.assign(year=lambda d: (d.index / 12.0)).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download percentiles (CSV)",
        data=export,
        file_name="monte_carlo_percentiles.csv",
        mime="text/csv",
        key="mc_export",
    )

    st.divider()

    summary = _summary_rows(nav_paths, ret_paths, horizon_years=horizon_years, initial=float(initial))
    render_table_report(
        summary,
        formats={
            "Terminal value": "{:,.0f}",
            "CAGR": "{:.2%}",
            "Simulated vol (ann.)": "{:.2%}",
        },
        numeric_cols=["Terminal value", "CAGR", "Simulated vol (ann.)"],
    )

    st.caption(
        "Percentiles are computed on the distribution of terminal values at the selected horizon. "
        "Simulated annualized volatility is the median of per-scenario annualized volatilities within each percentile band."
    )