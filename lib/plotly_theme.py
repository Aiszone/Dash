import plotly.io as pio

CURVO_BLUE = "#2563EB"
CURVO_GREEN = "#10B981"
CURVO_ORANGE = "#F97316"
CURVO_RED = "#DC2626"
CURVO_BG = "#FFFFFF"
CURVO_GRID = "#E5E7EB"
CURVO_TEXT = "#0F172A"


def register_curvo_finance_theme(template_name: str = "curvo_finance") -> str:
    base_layout = dict(
        font=dict(
            family="Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color=CURVO_TEXT,
        ),
        paper_bgcolor=CURVO_BG,
        plot_bgcolor=CURVO_BG,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor=CURVO_GRID,
            zeroline=False,
            ticks="outside",
            tickcolor=CURVO_GRID,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=CURVO_GRID,
            zeroline=False,
            ticks="outside",
            tickcolor=CURVO_GRID,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(229,231,235,0.7)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=CURVO_BLUE,
            font=dict(size=13, color=CURVO_TEXT),
        ),
        colorway=[
            CURVO_BLUE,
            CURVO_GREEN,
            CURVO_ORANGE,
            CURVO_RED,
        ],
    )

    pio.templates[template_name] = dict(layout=base_layout)
    return template_name