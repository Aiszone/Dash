from __future__ import annotations

import uuid
import numpy as np
import streamlit as st


def render_portfolio_builder(
    universe: dict[str, tuple[str, str]],
    default_portfolio: list[tuple[str, float]],
):
    st.markdown(
        """
        <style>
        div[data-testid="stContainer"]:has(.curvo-card-anchor) {
            background: #ffffff;
            border: 1px solid rgba(226,232,240,1);
            border-radius: 18px;
            padding: 18px 18px 10px 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .curvo-card-title {
            font-size: 18px;
            font-weight: 800;
            margin: 0 0 6px 0;
            color: #0f172a;
        }

        .curvo-card-subtitle {
            margin: 0 0 14px 0;
            color: rgba(15,23,42,0.6);
            font-size: 13px;
        }

        .curvo-divider {
            height: 1px;
            background: rgba(226,232,240,1);
            margin: 12px 0;
        }

        div[data-testid="stNumberInput"] input {
            background: #ffffff !important;
        }

        div[data-testid="stNumberInput"] button {
            background: #ffffff !important;
            border: 1px solid rgba(226,232,240,1) !important;
            border-radius: 10px !important;
            color: #0f172a !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: #f8fafc !important;
        }

        div[data-testid="stNumberInput"] {
            background: transparent !important;
        }

        div[data-testid="stNumberInput"] > div {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    asset_labels = list(universe.keys())

    def to_percent(x: float) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v <= 1.0:
            v = v * 100.0
        return float(max(0.0, min(100.0, v)))

    if "pb_rows" not in st.session_state:
        st.session_state.pb_rows = [
            {"id": uuid.uuid4().hex, "asset": a, "weight": to_percent(w)}
            for a, w in default_portfolio
        ]

    rows = st.session_state.pb_rows

    with st.container():
        st.markdown('<div class="curvo-card-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="curvo-card-title">Your portfolio</div>', unsafe_allow_html=True)
        st.markdown('<div class="curvo-card-subtitle">Positions and target weights</div>', unsafe_allow_html=True)

        h1, h2, h3 = st.columns([4, 2, 1])
        h1.markdown("**Asset**")
        h2.markdown("**Weight (%)**")
        h3.markdown("**Remove**")

        st.markdown('<div class="curvo-divider"></div>', unsafe_allow_html=True)

        for idx, row in enumerate(list(rows)):
            c1, c2, c3 = st.columns([4, 2, 1])

            with c1:
                sel = st.selectbox(
                    "",
                    options=asset_labels,
                    index=asset_labels.index(row["asset"]) if row["asset"] in asset_labels else 0,
                    key=f"asset_{row['id']}",
                    label_visibility="collapsed",
                )

            with c2:
                w = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    value=float(max(0.0, min(100.0, float(row["weight"])))),
                    key=f"weight_{row['id']}",
                    label_visibility="collapsed",
                )

            with c3:
                if st.button("✕", key=f"remove_{row['id']}"):
                    st.session_state.pb_rows.pop(idx)
                    st.rerun()

            row["asset"] = sel
            row["weight"] = float(w)

        st.markdown('<div class="curvo-divider"></div>', unsafe_allow_html=True)

        used = [r["asset"] for r in st.session_state.pb_rows]
        remaining = [a for a in asset_labels if a not in used] or asset_labels

        a1, a2, a3 = st.columns([4, 2, 1])

        with a1:
            new_asset = st.selectbox(
                "",
                options=remaining,
                index=0,
                key="pb_add_asset",
                label_visibility="collapsed",
            )

        with a2:
            new_weight = st.number_input(
                "",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                value=0.0,
                key="pb_add_weight",
                label_visibility="collapsed",
            )

        with a3:
            if st.button("＋", key="pb_add_btn"):
                st.session_state.pb_rows.append(
                    {"id": uuid.uuid4().hex, "asset": new_asset, "weight": float(new_weight)}
                )
                st.rerun()

    portfolio = [
        (r["asset"], float(r["weight"]) / 100.0)
        for r in st.session_state.pb_rows
        if float(r["weight"]) > 0
    ]

    total_weight = sum(w for _, w in portfolio)

    if len(portfolio) == 0:
        st.warning("Add at least one asset with weight > 0%.")
        return []

    if not np.isclose(total_weight, 1.0):
        st.warning(f"Total weight = {total_weight * 100:.1f}% (should be 100%).")

    return portfolio