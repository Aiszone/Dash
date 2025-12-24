# sections/efficient_frontier.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="all")
    r = prices.pct_change().dropna(how="any")
    return r


def _annual_stats(ret_m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu_m = ret_m.mean().values.astype(float)
    cov_m = ret_m.cov().values.astype(float)
    mu_a = mu_m * 12.0
    cov_a = cov_m * 12.0
    return mu_a, cov_a


def _portfolio_stats(w: np.ndarray, mu_a: np.ndarray, cov_a: np.ndarray, rf: float) -> tuple[float, float, float]:
    mu = float(w @ mu_a)
    sigma2 = float(w @ cov_a @ w)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    sharpe = (mu - rf) / sigma if sigma > 0 else -np.inf
    return mu, sigma, sharpe


def _dirichlet_sample(n_pts: int, n_assets: int, w_max: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(alpha=np.ones(n_assets), size=n_pts)
    if w_max >= 1.0:
        return W
    Wc = W.copy()
    for i in range(n_pts):
        w = Wc[i]
        for _ in range(30):
            too_big = w > w_max
            if not np.any(too_big):
                break
            excess = float(np.sum(w[too_big] - w_max))
            w[too_big] = w_max
            room = np.maximum(w_max - w, 0.0)
            room[too_big] = 0.0
            sroom = float(room.sum())
            if sroom <= 1e-12:
                w = np.minimum(w, w_max)
                w = w / w.sum()
                break
            w = w + (room / sroom) * excess
            w = np.minimum(w, w_max)
            w = w / w.sum()
        Wc[i] = w
    return Wc


def _regularize_cov(cov_a: np.ndarray, lam: float = 1e-8) -> np.ndarray:
    cov = cov_a.copy()
    cov.flat[:: cov.shape[0] + 1] += lam
    return cov


def _format_compact_comp(weights: np.ndarray, labels: list[str], top: int = 3) -> str:
    pairs = sorted([(float(w), lbl) for w, lbl in zip(weights, labels)], reverse=True, key=lambda x: x[0])
    parts = []
    for w, lbl in pairs[:top]:
        if w <= 1e-6:
            continue
        parts.append(f"{int(round(w * 100))}% {lbl}")
    return " • ".join(parts) if parts else "—"


def _make_frontier_chart(mu_all, sig_all, hover_comp, pt_curr, pt_min, pt_sharpe, pt_max) -> go.Figure:
    fig = go.Figure()

    fig.add_scatter(
        x=sig_all,
        y=mu_all,
        mode="markers",
        name="Sampled portfolios",
        marker=dict(size=5.5, color="rgba(37,99,235,0.55)"),
        customdata=hover_comp,
        hovertemplate=(
            "<span style='font-size:13px; font-weight:750;'>%{customdata}</span>"
            "<br><span style='font-size:12.5px;'>Risk: "
            "<span style='font-weight:800;'>%{x:.1%}</span>"
            "</span>"
            "<br><span style='font-size:12.5px;'>Return: "
            "<span style='font-weight:800;'>%{y:.1%}</span>"
            "</span><extra></extra>"
        ),
    )

    def add_marker(pt, name, color):
        if pt is None:
            return
        fig.add_scatter(
            x=[pt["sigma"]],
            y=[pt["mu"]],
            mode="markers",
            name=name,
            marker=dict(size=13, color=color, line=dict(width=1.5, color="white")),
            customdata=[pt["comp"]],
            hovertemplate=(
                "<span style='font-size:13px; font-weight:800;'>"
                + name
                + "</span>"
                "<br><span style='font-size:12.5px;'>%{customdata}</span>"
                "<br><span style='font-size:12.5px;'>Risk: "
                "<span style='font-weight:800;'>%{x:.1%}</span>"
                "</span>"
                "<br><span style='font-size:12.5px;'>Return: "
                "<span style='font-weight:800;'>%{y:.1%}</span>"
                "</span><extra></extra>"
            ),
        )

    add_marker(pt_curr, "Your portfolio", "#FACC15")
    add_marker(pt_min, "Minimum risk", "#DC2626")
    add_marker(pt_sharpe, "Max Sharpe", "#F97316")
    add_marker(pt_max, "Maximum return", "#10B981")

    fig.update_layout(
        title="Efficient frontier",
        template="plotly_white",
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(title="Risk (annualized)", tickformat=".1%"),
        yaxis=dict(title="Return (annualized)", tickformat=".1%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#2563EB",
            font=dict(size=13, color="#111827"),
            align="left",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(15,23,42,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.06)", zeroline=False)
    return fig


def _render_table(rows: list[dict]):
    css = """
    <style>
      .ef-card { border-radius: 14px; overflow: hidden; border: 1px solid #edf0f2; }
      .ef-head { background:#F6F7F8; font-weight:750; }
      .ef-row  { background:#FFFFFF; }
      .ef-row:nth-child(odd) { background:#FBFBFC; }
      .ef-cell { padding:12px 14px; font-size:14px; vertical-align: top; }
      .ef-right { text-align:right; white-space:nowrap; }
      .ef-table { width:100%; border-collapse: separate; border-spacing:0; }
      .ef-name { width: 180px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    def td(txt, cls=""):
        return "<td class=\"ef-cell {cls}\">{txt}</td>".format(cls=cls, txt=txt)

    head = (
        "<tr class='ef-head'>"
        + td("Portfolio", "ef-name")
        + td("Top holdings")
        + td("Return", "ef-right")
        + td("Risk", "ef-right")
        + td("Sharpe", "ef-right")
        + "</tr>"
    )

    body = []
    for r in rows:
        name_html = "<span style='color:{c}; font-weight:800;'>{t}</span>".format(c=r["color"], t=r["name"])
        body.append(
            "<tr class='ef-row'>"
            + td(name_html, "ef-name")
            + td(r["composition"])
            + td("{:.2f}%".format(r["mu"] * 100).replace(".", ","), "ef-right")
            + td("{:.2f}%".format(r["sigma"] * 100).replace(".", ","), "ef-right")
            + td("{:.2f}".format(r["sharpe"]) if np.isfinite(r["sharpe"]) else "—", "ef-right")
            + "</tr>"
        )

    html = "<div class='ef-card'><table class='ef-table'>" + head + "".join(body) + "</table></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_efficient_frontier(
    prices_monthly: pd.DataFrame,
    current_weights: pd.Series | np.ndarray,
    *,
    risk_free: float = 0.0,
    n_points: int = 1400,
    w_max: float = 1.0,
):
    st.header("Efficient frontier")
    st.caption("Random portfolios sampled from the asset universe.")

    if prices_monthly is None or prices_monthly.empty or prices_monthly.shape[1] < 2:
        st.info("Select at least 2 assets with enough history.")
        return
    if prices_monthly.shape[0] < 24:
        st.warning("At least 24 monthly points are required for a reliable frontier.")
        return

    labels = list(prices_monthly.columns)
    ret_m = _monthly_returns_from_prices(prices_monthly)
    if ret_m.shape[0] < 24:
        st.warning("Not enough monthly returns after cleaning.")
        return

    mu_a, cov_a = _annual_stats(ret_m)
    cov_a = _regularize_cov(cov_a, lam=1e-8)

    n_assets = len(labels)
    n_points = max(int(n_points), 400)
    rf = float(risk_free)

    W = _dirichlet_sample(n_points, n_assets, w_max=w_max, seed=42)

    mus = np.empty(n_points, dtype=float)
    sigs = np.empty(n_points, dtype=float)
    sharps = np.empty(n_points, dtype=float)
    hover_comp = []

    for i in range(n_points):
        mu_i, sig_i, sh_i = _portfolio_stats(W[i], mu_a, cov_a, rf)
        mus[i], sigs[i], sharps[i] = mu_i, sig_i, sh_i
        hover_comp.append(_format_compact_comp(W[i], labels, top=3))

    idx_min = int(np.argmin(sigs))
    idx_maxmu = int(np.argmax(mus))
    idx_sharp = int(np.argmax(sharps))

    def pt(idx, name, color):
        return dict(
            name=name,
            color=color,
            mu=float(mus[idx]),
            sigma=float(sigs[idx]),
            sharpe=float(sharps[idx]),
            w=W[idx],
            comp=_format_compact_comp(W[idx], labels, top=3),
        )

    pt_min = pt(idx_min, "Minimum risk", "#DC2626")
    pt_sharpe = pt(idx_sharp, "Max Sharpe", "#F97316")
    pt_max = pt(idx_maxmu, "Maximum return", "#10B981")

    if isinstance(current_weights, pd.Series):
        w_curr = current_weights.reindex(labels).fillna(0.0).values.astype(float)
    else:
        w_curr = np.asarray(current_weights, dtype=float)
        if w_curr.shape[0] != n_assets:
            w_curr = np.zeros(n_assets, dtype=float)
            w_curr[0] = 1.0

    w_curr = np.maximum(w_curr, 0.0)
    if w_curr.sum() > 0:
        w_curr = w_curr / w_curr.sum()

    mu_c, sig_c, shp_c = _portfolio_stats(w_curr, mu_a, cov_a, rf)
    pt_curr = dict(
        name="Your portfolio",
        color="#93C5FD",
        mu=float(mu_c),
        sigma=float(sig_c),
        sharpe=float(shp_c),
        w=w_curr,
        comp=_format_compact_comp(w_curr, labels, top=3),
    )

    fig = _make_frontier_chart(mus, sigs, hover_comp, pt_curr, pt_min, pt_sharpe, pt_max)
    st.plotly_chart(fig, use_container_width=True)

    df_pts = pd.DataFrame({"return": mus, "risk": sigs, "sharpe": sharps, "top3": hover_comp})
    st.download_button(
        "⬇️ Export sampled points (CSV)",
        data=df_pts.to_csv(index=False).encode("utf-8"),
        file_name="efficient_frontier_points.csv",
        mime="text/csv",
    )

    st.divider()

    _render_table(
        [
            dict(name=pt_curr["name"], color=pt_curr["color"], composition=pt_curr["comp"], mu=pt_curr["mu"], sigma=pt_curr["sigma"], sharpe=pt_curr["sharpe"]),
            dict(name=pt_min["name"], color=pt_min["color"], composition=pt_min["comp"], mu=pt_min["mu"], sigma=pt_min["sigma"], sharpe=pt_min["sharpe"]),
            dict(name=pt_sharpe["name"], color=pt_sharpe["color"], composition=pt_sharpe["comp"], mu=pt_sharpe["mu"], sigma=pt_sharpe["sigma"], sharpe=pt_sharpe["sharpe"]),
            dict(name=pt_max["name"], color=pt_max["color"], composition=pt_max["comp"], mu=pt_max["mu"], sigma=pt_max["sigma"], sharpe=pt_max["sharpe"]),
        ]
    )