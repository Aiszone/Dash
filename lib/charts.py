import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def make_pie(weights: pd.Series) -> go.Figure:
    df = pd.DataFrame({"Asset": weights.index, "Weight (%)": weights.values * 100.0})
    fig = px.pie(df, values="Weight (%)", names="Asset", hole=0.45)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=360, showlegend=False)
    return fig


def make_nav_chart(
    nav: pd.Series,
    extras: dict[str, pd.Series] | None = None,
    title: str = "Performance (base 10,000)",
) -> go.Figure:
    fig = go.Figure()

    ymin, ymax = 0.0, 12_000.0
    if nav is not None and not nav.empty:
        fig.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav.values,
                mode="lines",
                name="Portfolio",
                line=dict(width=2.3, color="#2563eb"),
            )
        )
        ymin = max(0.0, float(nav.min()) - 2000.0)
        ymax = float(nav.max()) + 2000.0

    if extras:
        for name, s in extras.items():
            if s is None or s.empty:
                continue
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(name)))
            ymin = min(ymin, float(s.min()) - 2000.0)
            ymax = max(ymax, float(s.max()) + 2000.0)

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=30, b=10),
        height=440,
        yaxis=dict(range=[ymin, ymax]),
    )
    return fig