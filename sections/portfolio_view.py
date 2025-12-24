import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from lib.portfolio import compute_stats_from_nav
from lib.ui import apply_plotly_style, kpi_row, render_table_report, section_header


def _fmt_number(x: float, decimals: int = 2) -> str:
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return ""


def fmt_money_usd(x: float) -> str:
    return f"${_fmt_number(x, 0)}"


def fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{_fmt_number(100 * x, decimals)}%"


def fmt_float(x: float, decimals: int = 2) -> str:
    return _fmt_number(x, decimals)


def _compute_drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.dropna().astype(float)
    if nav.empty:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    dd = nav / peak - 1.0
    dd.name = "Drawdown"
    return dd


def _max_drawdown_info(dd: pd.Series):
    if dd is None or dd.empty:
        return float("nan"), None, None, None

    trough_date = dd.idxmin()
    mdd_value = float(dd.loc[trough_date])

    dd_before = dd.loc[:trough_date]
    zeros = dd_before[dd_before == 0.0]
    peak_date = zeros.index[-1] if len(zeros) else dd_before.index[0]

    dd_after = dd.loc[trough_date:]
    rec = dd_after[dd_after == 0.0]
    recovery_date = rec.index[0] if len(rec) else None

    return mdd_value, peak_date, trough_date, recovery_date


def render_portfolio_view(
    prices_bm: pd.DataFrame,
    w: pd.Series,
    nav10k: pd.Series,
    start: str,
    end: str,
) -> None:
    section_header("Portfolio allocation", "Current weights used for the backtest.")
    pie_df = pd.DataFrame({"Asset": w.index, "Weight (%)": w.values * 100.0})
    fig_pie = px.pie(pie_df, values="Weight (%)", names="Asset", hole=0.45)
    fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=380, showlegend=False)
    apply_plotly_style(fig_pie)
    st.plotly_chart(fig_pie, use_container_width=True)

    section_header("Portfolio value", "NAV indexed to 10,000 USD.")
    fig_nav = go.Figure()
    fig_nav.add_trace(
        go.Scatter(
            x=nav10k.index,
            y=nav10k.values,
            mode="lines",
            line=dict(width=2.3),
            name="Portfolio",
        )
    )
    ymin, ymax = max(0.0, float(nav10k.min()) - 2000.0), float(nav10k.max()) + 2000.0
    fig_nav.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, yaxis=dict(range=[ymin, ymax]))
    apply_plotly_style(fig_nav)
    st.plotly_chart(fig_nav, use_container_width=True)

    dd = _compute_drawdown(nav10k)
    mdd, peak_dt, trough_dt, _rec_dt = _max_drawdown_info(dd)

    section_header("Drawdown", "Peak-to-trough decline from the running high.")
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100.0,
            mode="lines",
            line=dict(width=2.0),
            name="Drawdown (%)",
        )
    )
    fig_dd.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260, yaxis=dict(ticksuffix="%"))
    apply_plotly_style(fig_dd)
    st.plotly_chart(fig_dd, use_container_width=True)

    stats = compute_stats_from_nav(nav10k)
    invested = 10_000.0
    nav_val = float(stats.get("nav", float("nan")))
    cagr = float(stats.get("cagr", float("nan")))
    vol = float(stats.get("vol_ann", float("nan")))
    sharpe = float(stats.get("sharpe", float("nan")))

    if peak_dt is not None and trough_dt is not None:
        mdd_window = f"{peak_dt:%Y-%m-%d} → {trough_dt:%Y-%m-%d}"
    else:
        mdd_window = "—"

    section_header("Key metrics", "Summary statistics for the selected backtest window.")
    kpi_row(
        [
            {"label": "Invested", "value": fmt_money_usd(invested), "comment": "Base amount"},
            {"label": "NAV", "value": fmt_money_usd(nav_val), "comment": "End value"},
            {"label": "CAGR", "value": fmt_pct(cagr), "comment": "Annualized"},
            {"label": "Volatility", "value": fmt_pct(vol), "comment": "Annualized"},
            {"label": "Sharpe", "value": fmt_float(sharpe), "comment": "Rf = 0%"},
            {"label": "Max drawdown", "value": fmt_pct(mdd), "comment": mdd_window},
        ]
    )

    section_header("Performance attribution", "Cumulative returns and contributions over the backtest window.")

    cols = list(prices_bm.columns)

    perf_assets = []
    for col in cols:
        s = prices_bm[col].dropna()
        perf_assets.append(float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) >= 2 else float("nan"))

    contrib = []
    for col, p in zip(cols, perf_assets):
        wi = float(w.get(col, 0.0))
        contrib.append(wi * p)

    total_perf = float(np.nansum(contrib)) if len(contrib) else float("nan")

    share = []
    for c in contrib:
        if total_perf and not pd.isna(total_perf):
            share.append((c / total_perf) * 100.0)
        else:
            share.append(float("nan"))

    attrib_df = pd.DataFrame(
        {
            "Asset": cols,
            "Weight (%)": [float(w.get(col, 0.0)) * 100.0 for col in cols],
            "Total return (%)": [p * 100.0 for p in perf_assets],
            "Contribution (pp)": [c * 100.0 for c in contrib],
            "Share of total (%)": share,
        }
    ).sort_values("Contribution (pp)", ascending=False, ignore_index=True)

    render_table_report(
        attrib_df,
        formats={
            "Weight (%)": "{:.1f}%",
            "Total return (%)": "{:.2f}%",
            "Contribution (pp)": "{:.2f}",
            "Share of total (%)": "{:.1f}%",
        },
        numeric_cols=["Weight (%)", "Total return (%)", "Contribution (pp)", "Share of total (%)"],
    )