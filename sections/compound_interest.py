from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.compound import (
    DEFAULT_SEED,
    PathConfig,
    apply_lombard,
    cumulative_borrowed,
    cumulative_interest,
    simulate_compound_paths,
)
from lib.ui import apply_plotly_style, kpi_row, panel, section_header


def _fmt_years(x: int | None) -> str:
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)} years"
    return f"{float(x):.2f} years"


def _fmt_usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}".replace(",", " ")


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%".replace(".", ",")


def _first_hit(values: np.ndarray, threshold: float) -> int | None:
    if threshold is None or threshold <= 0 or values[-1] < threshold:
        return None
    idx = int(np.argmax(values >= threshold))
    return idx


def render() -> None:
    section_header("Compounding & Lombard loan", "Contribution paths with optional inflation adjustment and leverage overlay.")

    c1, c2, c3, c4, c5 = st.columns(5, vertical_alignment="center")
    v0 = c1.number_input("Initial capital (USD)", min_value=0.0, value=100_000.0, step=1_000.0, format="%.2f", key="ci_v0")
    years = c2.number_input("Duration (years)", min_value=1, max_value=50, value=25, step=1, key="ci_years")
    pmt = c3.number_input("Monthly contribution (USD)", min_value=0.0, value=1_000.0, step=100.0, format="%.2f", key="ci_pmt")
    seed = c4.number_input("Scenario seed", min_value=0, max_value=999_999, value=int(DEFAULT_SEED), step=1, key="ci_seed")
    infl_opts = [0, 1, 2, 3, 4, 5, 6]
    infl_pct = c5.selectbox("Inflation (%/year)", options=infl_opts, index=infl_opts.index(2), key="ci_infl")
    inflation = infl_pct / 100.0 if infl_pct > 0 else None

    section_header("Paths", "Enable up to three paths (CAGR + volatility).")

    cagr_vals = list(range(1, 26))
    vol_vals = list(range(0, 21))

    def path_row(name: str, default_cagr: int, default_vol: int, enabled: bool):
        a, b, c = st.columns([1.0, 1.0, 1.0], vertical_alignment="center")
        on = a.checkbox(name, value=enabled, key=f"ci_on_{name}")
        cagr_pct = b.selectbox("CAGR (%)", options=cagr_vals, index=cagr_vals.index(default_cagr), key=f"ci_cagr_{name}")
        vol_pct = c.selectbox("Volatility (%)", options=vol_vals, index=vol_vals.index(default_vol), key=f"ci_vol_{name}")
        return on, cagr_pct / 100.0, vol_pct / 100.0

    paths: list[PathConfig] = []
    e1, r1, s1 = path_row("Path 1", 8, 0, True)
    e2, r2, s2 = path_row("Path 2", 10, 10, False)
    e3, r3, s3 = path_row("Path 3", 12, 15, False)

    if e1:
        paths.append(PathConfig("Path 1", r1, s1))
    if e2:
        paths.append(PathConfig("Path 2", r2, s2))
    if e3:
        paths.append(PathConfig("Path 3", r3, s3))

    if not paths:
        st.info("Enable at least one path to run the simulation.")
        return

    target_cap = st.number_input(
        "Target value (USD)",
        min_value=0.0,
        value=1_000_000.0,
        step=50_000.0,
        format="%.2f",
        key="ci_target",
    )

    results = simulate_compound_paths(
        v0=float(v0),
        years=int(years),
        paths=paths,
        inflation=inflation,
        seed=int(seed),
        pmt_monthly=float(pmt),
    )

    invested_total = float(v0 + pmt * 12.0 * float(years))
    total_pmt = float(pmt * 12.0 * float(years))

    tab_comp, tab_lombard = st.tabs(["Compounding", "Lombard loan"])

    with tab_comp:
        show_real = inflation is not None

        fig = go.Figure()
        for label, df in results.items():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["V_nominal"],
                    mode="lines",
                    name=f"{label} — nominal",
                    hovertemplate="Year %{x}<br>Value: %{y:,.2f}<extra></extra>",
                )
            )
            if show_real:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["V_real"],
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"{label} — real",
                        hovertemplate="Year %{x}<br>Real value: %{y:,.2f}<extra></extra>",
                    )
                )

        if target_cap and target_cap > 0:
            fig.add_hline(y=float(target_cap), line_dash="dot", opacity=0.6)

        fig.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis_title="Years",
            yaxis_title="Value (USD)",
            hovermode="x unified",
        )
        apply_plotly_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        section_header("Path summary", "Key milestones and decomposition (table).")

        rows: list[dict] = []
        for label, df in results.items():
            v = df["V_nominal"].values
            final_nom = float(v[-1])
            t_double = _first_hit(v, 2.0 * float(v0))
            t_target = _first_hit(v, float(target_cap)) if target_cap and target_cap > 0 else None
            total_interest = final_nom - invested_total

            rows.append(
                {
                    "Path": label,
                    "Final value": _fmt_usd(final_nom),
                    "Time to double": _fmt_years(t_double),
                    "Time to target": _fmt_years(t_target),
                    "Total contributions": _fmt_usd(total_pmt),
                    "Interest earned (approx.)": _fmt_usd(total_interest),
                }
            )

        df_summary = pd.DataFrame(rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    with tab_lombard:
        section_header("Lombard settings", "Loan costs and target loan-to-value.")

        rate_opts = list(range(1, 16))
        ltv_opts = list(range(10, 81, 5))

        a, b, c, d = st.columns(4, vertical_alignment="center")
        rloan_pct = a.selectbox("Loan rate (%)", options=rate_opts, index=rate_opts.index(3), key="lb_rate")
        roll_years = b.selectbox("Refinancing interval (years)", options=[3, 4, 5], index=1, key="lb_roll")
        ltv_tgt_pct = c.selectbox("Target LTV (%)", options=ltv_opts, index=ltv_opts.index(40), key="lb_ltv_target")
        ltv0_pct = d.selectbox("Initial LTV (%)", options=ltv_opts, index=ltv_opts.index(40), key="lb_ltv0")

        r_loan = float(rloan_pct) / 100.0
        ltv_target = float(ltv_tgt_pct) / 100.0
        ltv0 = float(ltv0_pct) / 100.0

        apply_list = st.multiselect(
            "Apply the loan to these paths",
            options=list(results.keys()),
            default=list(results.keys())[:1],
            key="lb_apply",
        )

        if not apply_list:
            st.info("Select at least one path to apply leverage.")
            return

        lombard_outputs: dict[str, pd.DataFrame] = {}

        fig2 = go.Figure()
        for label in apply_list:
            df_path = results[label][["r_t", "V_nominal"]]
            dfL = apply_lombard(
                path_df=df_path,
                r_loan=r_loan,
                roll_years=int(roll_years),
                ltv_target=ltv_target,
                v0=float(v0),
                ltv0=ltv0,
                inflation=inflation,
                pmt_monthly=float(pmt),
            )
            lombard_outputs[label] = dfL

            fig2.add_trace(
                go.Scatter(
                    x=dfL.index,
                    y=dfL["P"],
                    mode="lines",
                    name=f"{label} — Portfolio",
                    hovertemplate="Year %{x}<br>P: %{y:,.2f}<extra></extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=dfL.index,
                    y=dfL["L"],
                    mode="lines",
                    name=f"{label} — Debt",
                    line=dict(dash="dash"),
                    hovertemplate="Year %{x}<br>L: %{y:,.2f}<extra></extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=dfL.index,
                    y=dfL["E"],
                    mode="lines",
                    name=f"{label} — Equity",
                    line=dict(dash="dot"),
                    hovertemplate="Year %{x}<br>E: %{y:,.2f}<extra></extra>",
                )
            )
            for rr in dfL.index[dfL["is_refi"]]:
                fig2.add_vline(x=int(rr), line_color="grey", opacity=0.25)

        fig2.update_layout(
            height=430,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis_title="Years",
            yaxis_title="Value (USD)",
            hovermode="x unified",
        )
        apply_plotly_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        for label, dfL in lombard_outputs.items():
            fig3.add_trace(
                go.Scatter(
                    x=dfL.index,
                    y=dfL["LTV"],
                    mode="lines",
                    name=f"{label} — LTV",
                    hovertemplate="Year %{x}<br>LTV: %{y:.2%}<extra></extra>",
                )
            )
        fig3.add_hline(y=float(ltv_target), line_dash="dot", opacity=0.6)
        fig3.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis_title="Years",
            yaxis_title="LTV",
            hovermode="x unified",
        )
        apply_plotly_style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        section_header("Leveraged path summary", "Debt, interest, and terminal state (table).")

        rowsL: list[dict] = []
        for label, dfL in lombard_outputs.items():
            debt_raised = float(cumulative_borrowed(dfL))
            interest_paid = float(cumulative_interest(dfL))

            p_final = float(dfL["P"].iloc[-1])
            l_final = float(dfL["L"].iloc[-1])
            e_final = float(dfL["E"].iloc[-1])
            ltv_fin = float(dfL["LTV"].iloc[-1])

            gain_net = e_final - (float(v0) + total_pmt)

            rowsL.append(
                {
                    "Path": label,
                    "Cumulative debt raised": _fmt_usd(debt_raised),
                    "Cumulative interest paid": _fmt_usd(interest_paid),
                    "Total contributions": _fmt_usd(total_pmt),
                    "Net gain (Equity − invested)": _fmt_usd(gain_net),
                    "Terminal portfolio": _fmt_usd(p_final),
                    "Terminal debt": _fmt_usd(l_final),
                    "Terminal equity": _fmt_usd(e_final),
                    "Terminal LTV": _fmt_pct(ltv_fin),
                }
            )

        dfL_summary = pd.DataFrame(rowsL)
        st.dataframe(dfL_summary, use_container_width=True, hide_index=True)