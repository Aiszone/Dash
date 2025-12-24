from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import fetch_series, to_monthly_bm
from lib.portfolio import nav_base_10k
from lib.ui import apply_plotly_style, panel, section_header


def _ensure_series(x, name: str = "NAV") -> pd.Series:
    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        raise TypeError("Expected a pandas Series or a 1-column DataFrame.")
    if s.name is None:
        s.name = name
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    return s.sort_index().dropna()


def _monthly_nav(s_price: pd.Series, label: str) -> pd.Series:
    sm = to_monthly_bm(s_price)
    if sm is None or sm.empty:
        return pd.Series(dtype=float, name=label)
    sm.name = label
    df = sm.to_frame(label)
    w = pd.Series({label: 1.0})
    nav = nav_base_10k(df, w)
    nav.name = label
    return nav


def _align_common_index(nav_port: pd.Series, benches: dict[str, pd.Series]) -> tuple[pd.Series, dict[str, pd.Series]]:
    idx = nav_port.index
    for s in benches.values():
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) < 2:
        return nav_port.iloc[0:0], {k: v.iloc[0:0] for k, v in benches.items()}
    return nav_port.loc[idx], {k: v.loc[idx] for k, v in benches.items()}


def _monthly_returns_from_nav(nav: pd.Series) -> pd.Series:
    r = nav.pct_change().dropna()
    r.name = f"ret_{nav.name}"
    return r


def _ann_cagr(nav: pd.Series) -> float:
    if len(nav) < 2:
        return np.nan
    n_years = len(nav) / 12.0
    if n_years <= 0:
        return np.nan
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_years) - 1.0)


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def _fmt_float(x: float | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def _stats_relative(port_nav: pd.Series, bench_nav: pd.Series) -> dict:
    pr = _monthly_returns_from_nav(port_nav)
    br = _monthly_returns_from_nav(bench_nav)
    df = pd.concat([pr, br], axis=1).dropna()
    if df.empty or df.shape[0] < 2:
        return dict(cum_alpha=np.nan, d_cagr=np.nan, te=np.nan, ir=np.nan, win=np.nan)

    port_ret = df.iloc[:, 0]
    bench_ret = df.iloc[:, 1]

    port_cum = float((1.0 + port_ret).prod() - 1.0)
    bench_cum = float((1.0 + bench_ret).prod() - 1.0)
    cum_alpha = port_cum - bench_cum

    d_cagr = _ann_cagr(port_nav) - _ann_cagr(bench_nav)

    active = port_ret - bench_ret
    te_ann = float(active.std(ddof=1) * np.sqrt(12.0)) if active.shape[0] > 1 else np.nan

    alpha_ann_simple = float(active.mean() * 12.0) if active.shape[0] > 0 else np.nan
    ir = float(alpha_ann_simple / te_ann) if (np.isfinite(te_ann) and te_ann != 0) else np.nan

    win = float((active > 0).mean()) if active.shape[0] > 0 else np.nan

    return dict(cum_alpha=cum_alpha, d_cagr=d_cagr, te=te_ann, ir=ir, win=win)


def _plot_navs(port_nav: pd.Series, benches: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()

    fig.add_scatter(
        x=port_nav.index,
        y=port_nav.values,
        mode="lines",
        name="Portfolio",
        line=dict(width=3),
        hovertemplate="<b>Portfolio</b><br>%{x|%b %Y}<br>NAV: %{y:,.0f}<extra></extra>",
    )

    for lbl, s in benches.items():
        fig.add_scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=lbl,
            line=dict(width=2),
            hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}}<br>NAV: %{{y:,.0f}}<extra></extra>",
        )

    fig.update_layout(
        title="NAV (base 10,000 USD)",
        xaxis_title="Date",
        yaxis_title="NAV",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    apply_plotly_style(fig)
    return fig


def _compute_capm_from_nav(
    port_nav: pd.Series,
    bench_nav: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 12,
) -> dict:
    pr = _monthly_returns_from_nav(port_nav)
    br = _monthly_returns_from_nav(bench_nav)

    df = pd.concat({"rp": pr, "rb": br}, axis=1).dropna()
    n = int(df.shape[0])
    if n < 3:
        return dict(beta=np.nan, alpha_period=np.nan, alpha_ann=np.nan, r2=np.nan, te_ann=np.nan, ir_capm=np.nan, n_obs=n, ex_rp=None, ex_rb=None)

    ex_rp = df["rp"] - rf
    ex_rb = df["rb"] - rf

    x = ex_rb.values
    y = ex_rp.values

    mean_x = float(x.mean())
    mean_y = float(y.mean())

    cov = float(np.mean((x - mean_x) * (y - mean_y)))
    var_x = float(np.mean((x - mean_x) ** 2))

    if var_x == 0:
        beta = np.nan
        alpha_period = np.nan
    else:
        beta = cov / var_x
        alpha_period = mean_y - beta * mean_x

    alpha_ann = float((1.0 + alpha_period) ** periods_per_year - 1.0) if np.isfinite(alpha_period) else np.nan

    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x > 0 and std_y > 0:
        corr = cov / (std_x * std_y)
        r2 = float(corr**2)
    else:
        r2 = np.nan

    active = df["rp"] - df["rb"]
    te_ann = float(active.std(ddof=1) * np.sqrt(periods_per_year)) if active.shape[0] > 1 else np.nan
    ir_capm = float(alpha_ann / te_ann) if (np.isfinite(te_ann) and te_ann != 0) else np.nan

    return dict(beta=beta, alpha_period=alpha_period, alpha_ann=alpha_ann, r2=r2, te_ann=te_ann, ir_capm=ir_capm, n_obs=n, ex_rp=ex_rp, ex_rb=ex_rb)


def _plot_capm_scatter(ex_rb: pd.Series, ex_rp: pd.Series, beta: float, alpha_period: float) -> go.Figure:
    fig = go.Figure()

    fig.add_scatter(
        x=ex_rb.values,
        y=ex_rp.values,
        mode="markers",
        name="Monthly observations",
        marker=dict(size=7, opacity=0.75),
        hovertemplate="Benchmark excess: %{x:.2%}<br>Portfolio excess: %{y:.2%}<extra></extra>",
    )

    if np.isfinite(beta) and np.isfinite(alpha_period):
        x_min = float(np.min(ex_rb.values))
        x_max = float(np.max(ex_rb.values))
        x_line = np.linspace(x_min, x_max, 60)
        y_line = alpha_period + beta * x_line
        fig.add_scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="CAPM fit",
            line=dict(width=2),
            hovertemplate="<extra></extra>",
        )

    fig.update_layout(
        title="CAPM scatter (excess returns)",
        xaxis_title="Benchmark (r_b - r_f)",
        yaxis_title="Portfolio (r_p - r_f)",
        height=480,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    apply_plotly_style(fig)
    return fig


def _table_card(data: dict[str, str]) -> None:
    st.dataframe(pd.DataFrame([data]), hide_index=True, use_container_width=True)


def render_benchmark_chart(
    UNIVERSE: dict,
    prices_portfolio_m: pd.DataFrame,
    nav_portfolio_10k: pd.Series,
    start_str: str,
    end_str: str,
) -> None:
    section_header("Benchmark", "Compare the portfolio vs benchmarks and run a CAPM regression.")

    nav_port = _ensure_series(nav_portfolio_10k, name="Portfolio")
    if nav_port.size < 2:
        st.warning("Not enough points to display benchmarks.")
        return

    bench_labels = list(UNIVERSE.keys())

    with panel("Benchmarks (max 3)", "Select up to 3 benchmarks to compare NAV and relative stats.", tight=True):
        default_multi = [lbl for lbl in bench_labels if lbl not in prices_portfolio_m.columns][:2]
        choices_multi = st.multiselect(
            "Benchmarks",
            options=bench_labels,
            default=default_multi,
            max_selections=3,
            label_visibility="collapsed",
        )

    benches_nav_multi: dict[str, pd.Series] = {}
    if choices_multi:
        for lbl in choices_multi:
            ticker, _ccy = UNIVERSE[lbl]
            s = fetch_series(ticker, start_str, end_str)
            nav_b = _monthly_nav(s, label=lbl)
            if not nav_b.empty:
                benches_nav_multi[lbl] = nav_b

    if not choices_multi or not benches_nav_multi:
        st.info("Select at least one benchmark to see the comparison.")
        return

    nav_port_multi, benches_nav_multi = _align_common_index(nav_port, benches_nav_multi)
    if nav_port_multi.empty:
        st.warning("Not enough overlapping dates with the selected benchmarks.")
        return

    section_header("NAV comparison", "All series rebased to 10,000 on the first common month.")
    st.plotly_chart(_plot_navs(nav_port_multi, benches_nav_multi), use_container_width=True)

    section_header("Relative stats", "Portfolio vs each benchmark on the common window.")
    for lbl, s in benches_nav_multi.items():
        stats = _stats_relative(nav_port_multi, s)
        st.markdown(f"### {lbl}")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            _table_card({"Cumulative outperformance": _fmt_pct(stats["cum_alpha"], 2)})
        with c2:
            _table_card({"ΔCAGR": _fmt_pct(stats["d_cagr"], 2)})
        with c3:
            _table_card({"Tracking error (ann.)": _fmt_pct(stats["te"], 2)})
        with c4:
            _table_card({"Information ratio": _fmt_float(stats["ir"], 2)})
        with c5:
            _table_card({"Win rate": _fmt_pct(stats["win"], 1)})

    section_header("CAPM", "Estimate beta/alpha vs one benchmark and optionally show the scatter.")

    default_capm_index = bench_labels.index(choices_multi[0]) if choices_multi else 0
    with panel("Primary benchmark", "Choose one benchmark for CAPM.", tight=True):
        primary_label = st.selectbox(
            "Primary benchmark",
            options=bench_labels,
            index=default_capm_index,
            label_visibility="collapsed",
        )

    ticker_capm, _ = UNIVERSE[primary_label]
    s_capm = fetch_series(ticker_capm, start_str, end_str)
    nav_b_capm = _monthly_nav(s_capm, label=primary_label)

    if nav_b_capm.empty:
        st.warning("No data available for the primary benchmark.")
        return

    capm = _compute_capm_from_nav(nav_port, nav_b_capm, rf=0.0, periods_per_year=12)
    if capm["n_obs"] < 3:
        st.info("Not enough points to estimate CAPM.")
        return

    st.markdown("### CAPM summary")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        _table_card({"Beta": _fmt_float(capm["beta"], 2)})
    with d2:
        _table_card({"Annualized alpha": _fmt_pct(capm["alpha_ann"], 2), "n (months)": str(capm["n_obs"])})
    with d3:
        _table_card({"R²": _fmt_float(capm["r2"], 2)})
    with d4:
        _table_card({"IR (alpha/TE)": _fmt_float(capm["ir_capm"], 2)})

    show_scatter = st.checkbox("Show CAPM scatter", value=True, key="show_capm_scatter")
    if show_scatter and capm["ex_rp"] is not None and capm["ex_rb"] is not None:
        st.plotly_chart(
            _plot_capm_scatter(
                ex_rb=capm["ex_rb"],
                ex_rp=capm["ex_rp"],
                beta=capm["beta"],
                alpha_period=capm["alpha_period"],
            ),
            use_container_width=True,
        )


render_benchmark = render_benchmark_chart
render_benchmark_view = render_benchmark_chart