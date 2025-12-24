from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.ui import apply_plotly_style, render_table_report, section_header


def _monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="all")
    return prices.pct_change().dropna(how="any")


def _pairwise_common_months(returns: pd.DataFrame) -> pd.DataFrame:
    cols = returns.columns
    n = len(cols)
    counts = pd.DataFrame(np.zeros((n, n), dtype=int), index=cols, columns=cols)
    for i, a in enumerate(cols):
        ai = returns[[a]].dropna()
        for j, b in enumerate(cols[i:], start=i):
            bj = returns[[b]].dropna()
            common = ai.index.intersection(bj.index)
            counts.iloc[i, j] = counts.iloc[j, i] = len(common)
    return counts


def _spectral_order(corr: pd.DataFrame) -> list[str]:
    r = corr.fillna(0.0).values
    r = np.clip(r, -0.9999, 0.9999)
    d = np.sqrt(2.0 * (1.0 - r))
    w = 1.0 / (1.0 + d)
    np.fill_diagonal(w, 0.0)
    deg = np.sum(w, axis=1)
    lap = np.diag(deg) - w
    vals, vecs = np.linalg.eigh(lap)
    idx = np.argsort(vals)
    if len(idx) < 2:
        return list(corr.columns)
    fiedler = vecs[:, idx[1]]
    order = np.argsort(fiedler)
    return [corr.columns[i] for i in order]


def _rolling_correlation(returns: pd.DataFrame, window: int) -> dict[pd.Timestamp, pd.DataFrame]:
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    if not window or window <= 0:
        return out
    for end in returns.index[window - 1 :]:
        start_idx = returns.index.get_loc(end) - window + 1
        if start_idx < 0:
            continue
        sub = returns.iloc[start_idx : start_idx + window].dropna()
        if len(sub) >= 6:
            out[end] = sub.corr(method="pearson")
    return out


def _plot_corr_heatmap(corr: pd.DataFrame, min_months: int, nobs: pd.DataFrame, title: str) -> go.Figure:
    c = corr.copy().mask(nobs < min_months)
    z = c.values.astype(float)
    texts = c.applymap(lambda x: "—" if (pd.isna(x) or not np.isfinite(x)) else f"{x:.2f}").values

    n = len(c.columns)
    height = max(420, int(52 * n))

    colorscale = [
        [0.00, "#b2182b"],
        [0.50, "#ffffff"],
        [1.00, "#1a9641"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(c.columns),
            y=list(c.index),
            zmin=-1.0,
            zmax=1.0,
            colorscale=colorscale,
            colorbar=dict(title="Correlation", tickformat=".2f"),
            text=texts,
            texttemplate="%{text}",
            textfont=dict(size=12),
            xgap=2,
            ygap=2,
            hovertemplate="A: %{y}<br>B: %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, constrain="domain"),
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
    )
    apply_plotly_style(fig)
    return fig


def _pairs_table(corr: pd.DataFrame, min_months: int, nobs: pd.DataFrame) -> pd.DataFrame:
    cols = list(corr.columns)
    rows: list[dict] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            n = int(nobs.loc[a, b])
            if n < min_months:
                continue
            r = float(corr.loc[a, b])
            rows.append({"Asset A": a, "Asset B": b, "Correlation": r, "Abs": abs(r)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["Abs", "Asset A", "Asset B"], ascending=[False, True, True], ignore_index=True)
    df = df.drop(columns=["Abs"])
    return df


def render_correlation_matrix(prices_monthly: pd.DataFrame, *, min_months: int = 36) -> None:
    section_header("Correlation", "Pearson correlation on monthly end-of-month returns.")

    if prices_monthly is None or prices_monthly.empty or prices_monthly.shape[1] < 2:
        st.info("Select at least 2 assets with sufficient history.")
        return
    if prices_monthly.shape[0] < min_months + 1:
        st.warning(f"At least {min_months + 1} monthly points are required for reliable correlations.")
        return

    returns = _monthly_returns_from_prices(prices_monthly)
    nobs = _pairwise_common_months(returns)
    corr = returns.corr(method="pearson")

    c1, c2 = st.columns([1.1, 1.0], vertical_alignment="center")
    with c1:
        order_mode = st.selectbox("Asset order", options=["Original", "Spectral (recommended)"], index=1, key="corr_order")
    with c2:
        roll_choice = st.selectbox("Rolling window", options=["None", "36 months", "60 months"], index=0, key="corr_roll")

    roll_window = 36 if roll_choice.startswith("36") else 60 if roll_choice.startswith("60") else None

    if order_mode.startswith("Spectral") and corr.shape[0] >= 3:
        ordered = _spectral_order(corr)
        corr_disp = corr.loc[ordered, ordered]
        nobs_disp = nobs.loc[ordered, ordered]
    else:
        corr_disp = corr
        nobs_disp = nobs

    fig = _plot_corr_heatmap(corr_disp, min_months=min_months, nobs=nobs_disp, title="Correlation matrix (monthly returns)")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download matrix (CSV)",
        data=corr_disp.to_csv().encode("utf-8"),
        file_name="correlation_matrix.csv",
        mime="text/csv",
        key="corr_export_matrix",
    )

    if roll_window:
        section_header("Rolling correlation", f"Latest window: {roll_window} months.")
        roll_map = _rolling_correlation(returns, window=int(roll_window))
        if not roll_map:
            st.info("Not enough windows to compute rolling correlations.")
        else:
            last_date = sorted(roll_map.keys())[-1]
            corr_last = roll_map[last_date]
            if order_mode.startswith("Spectral") and corr_last.shape[0] >= 3:
                corr_last = corr_last.loc[corr_disp.index, corr_disp.columns]

            fig_roll = _plot_corr_heatmap(
                corr_last,
                min_months=min_months,
                nobs=nobs_disp,
                title=f"Latest rolling window ({last_date.date()})",
            )
            st.plotly_chart(fig_roll, use_container_width=True)

            frames: list[pd.DataFrame] = []
            for dt, cm in roll_map.items():
                tmp = cm.copy()
                tmp["__date__"] = dt
                tmp = tmp.reset_index().melt(id_vars=["index", "__date__"], var_name="asset_b", value_name="corr")
                tmp = tmp.rename(columns={"index": "asset_a"})
                frames.append(tmp)
            df_roll = pd.concat(frames, ignore_index=True)

            st.download_button(
                "Download rolling correlations (CSV)",
                data=df_roll.to_csv(index=False).encode("utf-8"),
                file_name=f"rolling_correlation_{roll_window}m.csv",
                mime="text/csv",
                key="corr_export_roll",
            )

    st.divider()

    pairs = _pairs_table(corr_disp, min_months=min_months, nobs=nobs_disp)
    if pairs.empty:
        st.info(f"No pairs with at least {min_months} overlapping months.")
        return

    render_table_report(
        pairs,
        formats={"Correlation": "{:.2f}"},
        numeric_cols=["Correlation"],
    )

    st.download_button(
        "Download pairs (CSV)",
        data=pairs.to_csv(index=False).encode("utf-8"),
        file_name="correlation_pairs.csv",
        mime="text/csv",
        key="corr_export_pairs",
    )

    st.caption(f"Pairs with fewer than {min_months} overlapping months are excluded.")