import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


def _pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.{digits}f}%".replace(".", ",")


def render_returns_distribution(nav: pd.Series, start: str, end: str) -> None:
    if nav is None or nav.empty or len(nav) < 3:
        st.info("Not enough data points to analyze returns.")
        return

    nav = nav.sort_index()
    rets = nav.pct_change().dropna()
    if rets.empty:
        st.info("Not enough data to compute returns.")
        return

    freq = st.radio(
        "Frequency",
        options=["Annual", "Monthly"],
        index=1,
        horizontal=True,
        key="returns_freq",
    )

    def _make_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
        df = df.copy()
        df["color"] = np.where(df[y_col] >= 0, "#16A34A", "#DC2626")
        fig = go.Figure()
        fig.add_bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(color=df["color"]),
            customdata=np.stack([df["label"].values, df[y_col].values], axis=1),
            hovertemplate=(
                "<span style='font-size:13px; font-weight:750;'>%{customdata[0]}</span>"
                "<br><span style='font-size:12.5px;'>Return: "
                "<span style='font-weight:800;'>%{customdata[1]:+.2%}</span>"
                "</span><extra></extra>"
            ),
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis_title="Date",
            yaxis_title="Return",
            yaxis=dict(tickformat=".0%"),
            bargap=0.22,
            hovermode="x",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#2563EB",
                font=dict(size=13, color="#111827"),
                align="left",
            ),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(15,23,42,0.06)", zeroline=False)
        return fig

    def _best_worst(df: pd.DataFrame, label_col: str, ret_col: str):
        pos = df[df[ret_col] > 0].sort_values(ret_col, ascending=False).head(2).copy()
        neg = df[df[ret_col] < 0].sort_values(ret_col, ascending=True).head(2).copy()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Best periods**")
            if pos.empty:
                st.info("No positive periods.")
            else:
                t = pos[[label_col, ret_col]].copy()
                t["Return"] = t[ret_col].map(lambda x: _pct(float(x), 1))
                t = t.rename(columns={label_col: "Period"})[["Period", "Return"]]
                st.table(t.set_index("Period"))

        with c2:
            st.markdown("**Worst periods**")
            if neg.empty:
                st.info("No negative periods.")
            else:
                t = neg[[label_col, ret_col]].copy()
                t["Return"] = t[ret_col].map(lambda x: _pct(float(x), 1))
                t = t.rename(columns={label_col: "Period"})[["Period", "Return"]]
                st.table(t.set_index("Period"))

    if freq == "Annual":
        df = pd.DataFrame({"ret": rets})
        df["year"] = df.index.year
        annual = df.groupby("year")["ret"].apply(lambda r: (1.0 + r).prod() - 1.0).reset_index()
        annual = annual.rename(columns={"year": "x", "ret": "ret"})
        if annual.empty:
            st.info("Not enough data to compute annual returns.")
            return
        annual["label"] = annual["x"].astype(int).astype(str)

        fig = _make_bar(annual, x_col="x", y_col="ret", title="Annual returns")
        fig.update_xaxes(type="category", title_text="Year")
        st.plotly_chart(fig, use_container_width=True)

        _best_worst(annual, label_col="label", ret_col="ret")

    else:
        df = pd.DataFrame({"ret": rets})
        df["date"] = df.index
        df["label"] = df["date"].dt.strftime("%b %Y")
        if df.empty:
            st.info("Not enough data to compute monthly returns.")
            return

        fig = _make_bar(df, x_col="date", y_col="ret", title="Monthly returns")
        fig.update_xaxes(title_text="Month")
        st.plotly_chart(fig, use_container_width=True)

        _best_worst(df, label_col="label", ret_col="ret")