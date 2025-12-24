from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import streamlit as st

from lib.portfolio import build_prices_usd_monthly, nav_base_10k, weights_vector
from lib.ui import inject_global_css, page_header

from sections.correlation_matrix import render_correlation_matrix
from sections.efficient_frontier import render_efficient_frontier
from sections.factor_exposure import render_factor_exposure
from sections.monte_carlo_simulation import render_monte_carlo
from sections.portfolio_builder import render_portfolio_builder
from sections.portfolio_view import render_portfolio_view
from sections.returns_distribution import render_returns_distribution
from sections.risk_stress import render_risk_stress

try:
    from sections import compound_interest as compound_section
except Exception:
    compound_section = None

try:
    import sections.benchmark_chart as benchmark_mod
except Exception:
    benchmark_mod = None


st.set_page_config(page_title="Backtest Dashboard (USD)", layout="wide")
inject_global_css()


def _resolve_benchmark_renderer():
    if benchmark_mod is None:
        return None
    for name in ("render_benchmark_chart", "render_benchmark", "render_benchmark_view"):
        fn = getattr(benchmark_mod, name, None)
        if callable(fn):
            return fn
    return None


_BENCH_RENDER = _resolve_benchmark_renderer()


def _render_benchmark_fallback():
    st.warning(
        "Benchmark tab is enabled but no renderer function was found in `sections/benchmark_chart.py`.\n\n"
        "Define one of: `render_benchmark_chart`, `render_benchmark`, `render_benchmark_view`."
    )


def _normalize_selection(selection_raw) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if not selection_raw:
        return out
    for item in selection_raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                lbl = str(item[0])
                w = float(item[1])
            except Exception:
                continue
            out.append((lbl, w))
    return out


def _select_backtest_months_from_prices(prices: pd.DataFrame):
    if prices is None or prices.empty:
        st.error("No price data available to select a backtest period.")
        st.stop()

    df = prices.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index().dropna(how="any")
    if df.empty:
        st.error("Price history is empty after cleaning.")
        st.stop()

    months = df.index.to_period("M").unique().sort_values()
    if len(months) < 2:
        st.error(
            "Not enough overlapping monthly history across selected assets.\n\n"
            "Try removing the newest/short-history asset."
        )
        st.caption(f"Common months found: {len(months)}")
        st.stop()

    cA, cB = st.columns([1, 4])
    with cA:
        if st.button("Reset dates", use_container_width=True):
            st.session_state.pop("bt_start_month", None)
            st.session_state.pop("bt_end_month", None)
            st.rerun()
    with cB:
        st.write("")

    if "bt_start_month" in st.session_state and st.session_state["bt_start_month"] not in months:
        st.session_state["bt_start_month"] = months[0]
    if "bt_end_month" in st.session_state and st.session_state["bt_end_month"] not in months:
        st.session_state["bt_end_month"] = months[-1]

    col1, col2 = st.columns(2)
    with col1:
        start_period = st.selectbox(
            "Start month",
            options=months,
            format_func=lambda p: p.strftime("%B %Y"),
            index=0,
            key="bt_start_month",
        )
    with col2:
        end_period = st.selectbox(
            "End month",
            options=months,
            format_func=lambda p: p.strftime("%B %Y"),
            index=len(months) - 1,
            key="bt_end_month",
        )

    if start_period > end_period:
        st.warning("Start month was after End month — adjusted automatically.")
        start_period = months[0]
        end_period = months[-1]
        st.session_state["bt_start_month"] = start_period
        st.session_state["bt_end_month"] = end_period

    if start_period == end_period:
        i = int(np.where(months == end_period)[0][0])
        end_period = months[min(i + 1, len(months) - 1)]
        st.session_state["bt_end_month"] = end_period
        st.info("End month was extended by 1 month to ensure enough data points.")

    mask = (df.index.to_period("M") >= start_period) & (df.index.to_period("M") <= end_period)
    df_sel = df.loc[mask]

    if len(df_sel) < 2:
        st.warning("Selected period too short — falling back to full available range.")
        df_sel = df.copy()

    start_date = df_sel.index.min()
    end_date = df_sel.index.max()
    st.caption(f"Effective period: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")

    return df_sel, start_date, end_date


UNIVERSE = {
    "S&P 500 (SPY)": ("SPY", "USD"),
    "Nasdaq 100 (QQQ)": ("QQQ", "USD"),
    "MSCI World (VT)": ("VT", "USD"),
    "Emerging Markets (EEM)": ("EEM", "USD"),
    "Europe (VGK)": ("VGK", "USD"),
    "India (INDA)": ("INDA", "USD"),
    "China (MCHI)": ("MCHI", "USD"),
    "US Treasury 20Y (TLT)": ("TLT", "USD"),
    "Gold (GLD)": ("GLD", "USD"),
    "Silver (SLV)": ("SLV", "USD"),
    "Oil (USO)": ("USO", "USD"),
    "Bitcoin (BTC-USD)": ("BTC-USD", "USD"),
    "Ethereum (ETH-USD)": ("ETH-USD", "USD"),
}

DEFAULT_PORTFOLIO = [
    ("S&P 500 (SPY)", 60.0),
    ("Gold (GLD)", 20.0),
    ("Nasdaq 100 (QQQ)", 20.0),
]


page_header(
    "Backtest Dashboard",
    "USD-only • NAV base 10,000 • Portfolio builder + backtest window + risk & factor tools",
)


if "selection" not in st.session_state:
    st.session_state.selection = DEFAULT_PORTFOLIO

show_builder = st.toggle("Show portfolio builder", value=True) if hasattr(st, "toggle") else st.checkbox(
    "Show portfolio builder", value=True
)

if show_builder:
    selection_raw = render_portfolio_builder(UNIVERSE, DEFAULT_PORTFOLIO)
    st.session_state.selection = selection_raw
else:
    selection_raw = st.session_state.selection
    st.caption("Portfolio builder is hidden.")


try:
    selection = _normalize_selection(selection_raw)
    if not selection:
        st.warning("Add at least one asset with a weight > 0%.")
        st.stop()

    fetch_start = "2000-01-01"
    fetch_end = pd.Timestamp.today().strftime("%Y-%m-%d")

    with st.spinner("Downloading data..."):
        prices_all = build_prices_usd_monthly(selection, UNIVERSE, fetch_start, fetch_end)

    if prices_all is None or prices_all.empty:
        st.error("No overlapping price history across the selected assets.")
        st.stop()

    prices, eff_start, eff_end = _select_backtest_months_from_prices(prices_all)
    eff_start_str = eff_start.strftime("%Y-%m-%d")
    eff_end_str = eff_end.strftime("%Y-%m-%d")

    w = weights_vector(selection)
    nav10k = nav_base_10k(prices, w)
    nav10k.name = "NAV"

    if nav10k is None or nav10k.empty or len(nav10k) < 2:
        st.error("NAV could not be computed (not enough overlapping points after filtering).")
        st.stop()

    tabs = st.tabs(
        [
            "Portfolio",
            "Benchmark",
            "Returns",
            "Monte Carlo",
            "Frontier",
            "Correlation",
            "Risk & Stress",
            "Factors",
            "Compound Interest",
        ]
    )

    with tabs[0]:
        render_portfolio_view(prices, w, nav10k, eff_start_str, eff_end_str)

    with tabs[1]:
        if _BENCH_RENDER is None:
            _render_benchmark_fallback()
        else:
            _BENCH_RENDER(UNIVERSE, prices, nav10k, eff_start_str, eff_end_str)

    with tabs[2]:
        render_returns_distribution(nav=nav10k, start=eff_start_str, end=eff_end_str)

    with tabs[3]:
        render_monte_carlo(nav10k, start=eff_start_str, end=eff_end_str)

    with tabs[4]:
        render_efficient_frontier(
            prices_monthly=prices,
            current_weights=w,
            risk_free=0.00,
            n_points=1200,
            w_max=1.0,
        )

    with tabs[5]:
        render_correlation_matrix(prices, min_months=12)

    with tabs[6]:
        render_risk_stress(prices, w, nav10k, UNIVERSE, selection)

    with tabs[7]:
        render_factor_exposure(nav10k, start=eff_start_str, end=eff_end_str)

    with tabs[8]:
        if compound_section is None or not hasattr(compound_section, "render"):
            st.warning("Compound Interest section is unavailable.")
        else:
            compound_section.render()

except Exception as e:
    st.error(f"Backtest error: {type(e).__name__}: {e}")
    st.code("".join(traceback.format_exc()))