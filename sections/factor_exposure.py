from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import fetch_series, to_monthly_bm
from lib.ui import apply_plotly_style, kpi_row, panel, render_table_report, section_header


FACTOR_UNIVERSE = {
    "US Market": "SPY",
    "Europe": "VGK",
    "Emerging Markets": "EEM",
    "Tech / Growth": "QQQ",
    "Long Rates (US Treasuries 20y+)": "TLT",
    "Gold": "GLD",
    "Oil": "USO",
    "Crypto (Bitcoin)": "BTC-USD",
}


def _portfolio_returns_from_nav(nav10k: pd.Series) -> pd.Series:
    r = nav10k.pct_change().dropna()
    if r.empty:
        raise ValueError("Unable to compute portfolio returns (NAV series too short).")
    r.name = "Portfolio"
    return r


def _factor_returns(ticker: str, start: str, end: str) -> pd.Series:
    s = fetch_series(ticker, start, end)
    sm = to_monthly_bm(s)
    if sm is None or sm.empty:
        return pd.Series(dtype=float, name=ticker)
    r = sm.pct_change().dropna()
    r.name = ticker
    return r


def _align(y: pd.Series, factors: dict[str, pd.Series]) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.concat([y] + list(factors.values()), axis=1, join="inner").dropna(how="any")
    if df.shape[0] < 24:
        raise ValueError("Common history is too short for a robust factor regression (< 24 monthly points).")
    y_aligned = df.iloc[:, 0]
    x_aligned = df.iloc[:, 1:]
    x_aligned.columns = list(factors.keys())
    return y_aligned, x_aligned


def _ols(y: pd.Series, x: pd.DataFrame) -> tuple[pd.Series, float, pd.Series]:
    yv = y.values.reshape(-1, 1)
    xm = x.values
    n, k = xm.shape

    x_design = np.column_stack([np.ones(n), xm])
    coef, residuals, _, _ = np.linalg.lstsq(x_design, yv, rcond=None)
    coef = coef.flatten()

    alpha = float(coef[0])
    betas = coef[1:]

    if residuals.size > 0:
        sse = float(residuals[0])
    else:
        y_hat = x_design @ coef.reshape(-1, 1)
        sse = float(np.sum((yv - y_hat) ** 2))

    dof = max(n - (k + 1), 1)
    mse = sse / dof

    xtx = x_design.T @ x_design
    xtx_inv = np.linalg.inv(xtx)
    se = np.sqrt(np.diag(xtx_inv) * mse)
    tstats = coef / se

    betas_s = pd.Series(betas, index=x.columns, name="beta")
    tstats_s = pd.Series(tstats[1:], index=x.columns, name="tstat")
    return betas_s, alpha, tstats_s


def _r2(y: pd.Series, x: pd.DataFrame, betas: pd.Series, alpha: float) -> float:
    yv = y.values
    y_hat = alpha + x.values @ betas.values
    sst = float(np.sum((yv - yv.mean()) ** 2))
    sse = float(np.sum((yv - y_hat) ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - sse / sst


def _cum_contrib(betas: pd.Series, x: pd.DataFrame) -> pd.Series:
    out: dict[str, float] = {}
    for f in x.columns:
        r = x[f].dropna()
        out[f] = np.nan if r.empty else float(betas.get(f, 0.0) * ((1.0 + r).prod() - 1.0))
    return pd.Series(out, name="contrib")


def _fmt_pct(x: float | None, d: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{d}f}%".replace(".", ",")


def _fmt_float(x: float | None, d: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def _bar_betas(betas: pd.Series) -> go.Figure:
    b = betas.sort_values(ascending=False)
    fig = go.Figure()
    fig.add_bar(
        x=b.index.tolist(),
        y=b.values.tolist(),
        customdata=np.array(b.values),
        hovertemplate="Factor: %{x}<br>Beta: %{customdata:.2f}<extra></extra>",
        name="Betas",
    )
    fig.update_layout(
        title="Factor betas",
        xaxis_title="Factor",
        yaxis_title="Beta",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    apply_plotly_style(fig)
    return fig


def _radar_betas(betas: pd.Series) -> go.Figure:
    b = betas.sort_values(ascending=False)
    theta = b.index.tolist()
    r_vals = b.values.tolist()
    fig = go.Figure(
        data=go.Scatterpolar(
            r=r_vals + [r_vals[0]],
            theta=theta + [theta[0]],
            fill="toself",
            hovertemplate="Factor: %{theta}<br>Beta: %{r:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Factor map",
        polar=dict(radialaxis=dict(visible=True, tickformat=".2f")),
        showlegend=False,
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    apply_plotly_style(fig)
    return fig


def render_factor_exposure(nav10k: pd.Series, start: str | None = None, end: str | None = None) -> None:
    section_header("Factor exposure", "Multi-factor regression using ETF proxies and monthly portfolio returns.")

    if nav10k is None or nav10k.empty:
        st.warning("No NAV data available for factor analysis.")
        return

    nav = nav10k.dropna().copy()
    if not isinstance(nav.index, pd.DatetimeIndex):
        nav.index = pd.to_datetime(nav.index)

    if len(nav) < 24:
        st.warning("Portfolio history is too short for factor analysis (< 24 points).")
        return

    start = start or nav.index.min().strftime("%Y-%m-%d")
    end = end or nav.index.max().strftime("%Y-%m-%d")

    try:
        r_port = _portfolio_returns_from_nav(nav)
    except Exception as e:
        st.error(f"Unable to compute portfolio returns: {e}")
        return

    factor_returns: dict[str, pd.Series] = {}
    for name, ticker in FACTOR_UNIVERSE.items():
        try:
            r_f = _factor_returns(ticker, start, end)
        except Exception:
            continue
        if r_f is None or r_f.empty:
            continue
        r_f.name = name
        factor_returns[name] = r_f

    if not factor_returns:
        st.error("Unable to retrieve factor data.")
        return

    try:
        y, x = _align(r_port, factor_returns)
        betas, alpha, tstats = _ols(y, x)
        r2 = _r2(y, x, betas, alpha)
    except Exception as e:
        st.error(f"Factor regression failed: {e}")
        return

    contrib = _cum_contrib(betas, x)
    total_port = float((1.0 + y).prod() - 1.0)

    beta_mkt = float(betas.get("US Market", betas.iloc[0] if len(betas) else np.nan))
    try:
        alpha_ann = (1.0 + float(alpha)) ** 12 - 1.0
    except Exception:
        alpha_ann = np.nan

    with panel("Summary", "Key regression metrics", tight=True):
        kpi_row(
            [
                {"label": "US equity beta", "value": _fmt_float(beta_mkt, 2), "comment": ""},
                {"label": "Alpha (annualized, approx.)", "value": _fmt_pct(alpha_ann, 2), "comment": ""},
                {"label": "Regression R²", "value": _fmt_pct(r2, 1), "comment": ""},
            ]
        )

    st.plotly_chart(_bar_betas(betas), use_container_width=True)

    if len(betas) >= 3:
        with panel("Factor map", "Radar view of betas", tight=False):
            st.plotly_chart(_radar_betas(betas), use_container_width=True)

    df = pd.DataFrame(
        {
            "Factor": betas.index,
            "Beta": betas.values,
            "t-stat": tstats.reindex(betas.index).values,
            "Contribution (pp)": contrib.reindex(betas.index).values * 100.0,
            "Share of total (%)": [
                (c / total_port) * 100.0 if (np.isfinite(total_port) and total_port != 0 and np.isfinite(c)) else np.nan
                for c in contrib.reindex(betas.index).values
            ],
        }
    ).sort_values("Contribution (pp)", ascending=False, ignore_index=True)

    with panel("Details", "Betas, t-stats and approximate contributions", tight=False):
        render_table_report(
            df,
            formats={
                "Beta": "{:.2f}",
                "t-stat": "{:.2f}",
                "Contribution (pp)": "{:.2f}",
                "Share of total (%)": "{:.1f}%",
            },
            numeric_cols=["Beta", "t-stat", "Contribution (pp)", "Share of total (%)"],
        )