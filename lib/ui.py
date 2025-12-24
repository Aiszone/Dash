from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
import streamlit as st

UI = {
    "primary": "#2563EB",
    "accent": "#EA580C",
    "text": "#111827",
    "muted": "#6B7280",
    "bg": "#F3F4F6",
    "panel": "#FFFFFF",
    "border": "#E5E7EB",
    "shadow": "rgba(15,23,42,0.08)",
    "shadow_strong": "rgba(15,23,42,0.12)",
    "white": "#FFFFFF",
    "success": "#16A34A",
    "warning": "#EA580C",
    "danger": "#DC2626",
    "yellow": "#F59E0B",
}


def inject_global_css() -> None:
    st.markdown(
        f"""
<style>
:root {{
  --curvo-accent: {UI["accent"]};
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
  box-shadow: none !important;
  border-bottom: none !important;
}}

div[data-testid="stToolbar"] {{
  background: transparent !important;
}}

div[data-testid="stDecoration"] {{
  display: none !important;
}}

.stApp {{
  background: {UI["bg"]};
}}

.block-container {{
  padding-top: 0.55rem !important;
  padding-bottom: 2rem !important;
}}

.ui-page-title {{
  font-size: 40px;
  font-weight: 850;
  letter-spacing: -0.02em;
  line-height: 1.05;
  color: {UI["text"]};
  margin: 0 0 6px 0;
}}
.ui-page-subtitle {{
  font-size: 16px;
  color: {UI["muted"]};
  margin: 0 0 18px 0;
}}

.ui-section {{ margin-top: 18px; margin-bottom: 6px; }}
.ui-section-title {{
  font-size: 18px;
  font-weight: 850;
  color: {UI["text"]};
  margin: 0 0 6px 0;
}}
.ui-section-subtitle {{
  font-size: 13px;
  color: {UI["muted"]};
  margin: 0 0 10px 0;
}}
.ui-accent {{
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  background: {UI["primary"]};
  margin-right: 10px;
  transform: translateY(-1px);
}}

.ui-panel {{
  background: {UI["panel"]};
  border: 1px solid {UI["border"]};
  border-radius: 16px;
  padding: 16px 18px;
  margin: 14px 0;
  box-shadow: 0 18px 36px {UI["shadow"]};
}}
.ui-panel-tight {{
  padding: 12px 14px;
}}
.ui-panel-head {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}}
.ui-panel-dot {{
  width: 12px;
  height: 12px;
  border-radius: 4px;
  background: {UI["primary"]};
  flex: 0 0 12px;
}}
.ui-panel-title {{
  font-size: 18px;
  font-weight: 850;
  color: {UI["text"]};
  margin: 0;
}}
.ui-panel-subtitle {{
  font-size: 13px;
  color: {UI["muted"]};
  margin: 8px 0 0 22px;
}}

.ui-kpi {{
  background: {UI["panel"]};
  border: 1px solid {UI["border"]};
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 16px 30px {UI["shadow"]};
}}
.ui-kpi-label {{
  font-size: 10px;
  color: {UI["muted"]};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}}
.ui-kpi-value {{
  font-size: 24px;
  font-weight: 900;
  color: {UI["text"]};
  margin-bottom: 4px;
}}
.ui-kpi-comment {{
  font-size: 12px;
  color: {UI["muted"]};
}}

div[data-testid="stPlotlyChart"] {{
  background: {UI["panel"]};
  border: 1px solid {UI["border"]};
  border-radius: 18px;
  padding: 10px 12px;
  box-shadow: 0 18px 36px {UI["shadow_strong"]};
  margin: 10px 0 18px 0;
}}

.js-plotly-plot .hoverlayer .hovertext rect {{
  rx: 10px;
  ry: 10px;
  filter: drop-shadow(0px 12px 18px rgba(15,23,42,0.18));
  stroke: {UI["primary"]} !important;
  stroke-width: 1px !important;
}}
.js-plotly-plot .hoverlayer .hovertext text {{
  font-size: 13.5px !important;
  font-weight: 650 !important;
}}

.ui-report-wrap {{
  display: flex;
  justify-content: center;
  width: 100%;
  margin: 10px 0 18px 0;
}}
.ui-report-card {{
  background: {UI["panel"]};
  border: 1px solid {UI["border"]};
  border-radius: 16px;
  box-shadow: 0 18px 36px {UI["shadow_strong"]};
  padding: 10px 12px;
  width: fit-content;
  max-width: 96%;
  overflow-x: auto;
}}
table.ui-report {{
  border-collapse: collapse;
  table-layout: auto;
  width: auto;
  font-size: 13px;
  white-space: nowrap;
}}
table.ui-report thead th {{
  text-align: left;
  font-size: 12px;
  color: {UI["muted"]};
  font-weight: 850;
  padding: 10px 12px;
  border-bottom: 1px solid {UI["border"]};
  background: #EEF2F7;
}}
table.ui-report tbody td {{
  padding: 9px 12px;
  border-bottom: 1px solid {UI["border"]};
  color: {UI["text"]};
  vertical-align: middle;
}}
table.ui-report tbody tr:hover {{ background: rgba(37,99,235,0.08); }}
table.ui-report td.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}

.ui-badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
  border: 1px solid rgba(0,0,0,0.06);
}}
.ui-badge-blue {{ background: rgba(37,99,235,0.10); color: {UI["primary"]}; }}
.ui-badge-green {{ background: rgba(22,163,74,0.12); color: {UI["success"]}; }}
.ui-badge-red {{ background: rgba(220,38,38,0.10); color: {UI["danger"]}; }}
.ui-badge-yellow {{ background: rgba(245,158,11,0.14); color: {UI["yellow"]}; }}

.ui-divider {{
  height: 1px;
  width: 100%;
  background: {UI["border"]};
  margin: 18px 0 18px 0;
  opacity: 0.9;
}}

.ui-head {{ font-weight: 800; color: {UI["text"]}; margin: 0 0 6px 0; }}
.ui-actions {{ display: flex; justify-content: flex-end; }}

.ui-row-shell {{
  background: {UI["panel"]};
  border: 1px solid {UI["border"]};
  border-radius: 16px;
  box-shadow: 0 14px 26px {UI["shadow"]};
  padding: 10px 12px;
  margin: 10px 0;
}}

button {{ border-radius: 12px !important; }}

div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"] {{
  background-color: transparent !important;
}}

div[data-baseweb="select"] {{
  background-color: {UI["white"]} !important;
  border-radius: 12px !important;
}}
div[data-baseweb="select"] > div {{
  background-color: {UI["white"]} !important;
}}
div[data-baseweb="select"] [role="combobox"] {{
  background-color: {UI["white"]} !important;
}}
div[data-baseweb="popover"] {{
  background-color: {UI["white"]} !important;
}}

div[data-baseweb="select"] [role="combobox"] span {{
  color: var(--curvo-accent) !important;
  font-weight: 850 !important;
}}

div[data-baseweb="select"] > div:focus-within {{
  border-color: rgba(234,88,12,0.70) !important;
  box-shadow: 0 0 0 3px rgba(234,88,12,0.18) !important;
}}

div[data-baseweb="tag"] {{
  background: var(--curvo-accent) !important;
  border: 1px solid rgba(234,88,12,0.20) !important;
  border-radius: 10px !important;
}}
div[data-baseweb="tag"] span {{
  color: #ffffff !important;
  font-weight: 750 !important;
}}
div[data-baseweb="tag"] svg {{
  color: #ffffff !important;
  fill: #ffffff !important;
}}

ul[role="listbox"] li[aria-selected="true"] {{
  background: rgba(234,88,12,0.10) !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"<div class='ui-page-title'>{html.escape(title)}</div>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<div class='ui-page-subtitle'>{html.escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown("<div class='ui-section'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='ui-section-title'><span class='ui-accent'></span>{html.escape(title)}</div>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<div class='ui-section-subtitle'>{html.escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


@contextmanager
def panel(title: str | None = None, subtitle: str | None = None, *, tight: bool = False) -> Iterator[None]:
    cls = "ui-panel ui-panel-tight" if tight else "ui-panel"
    head = ""
    if title:
        head = f"""
<div class="ui-panel-head">
  <div class="ui-panel-dot"></div>
  <div class="ui-panel-title">{html.escape(title)}</div>
</div>
"""
    sub = ""
    if subtitle:
        sub = f"""<div class="ui-panel-subtitle">{html.escape(subtitle)}</div>"""
    st.markdown(f"""<div class="{cls}">{head}{sub}""", unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "blue") -> None:
    cls = {
        "blue": "ui-badge-blue",
        "green": "ui-badge-green",
        "red": "ui-badge-red",
        "yellow": "ui-badge-yellow",
    }.get(kind, "ui-badge-blue")
    st.markdown(
        f"<span class='ui-badge {cls}'>{html.escape(text)}</span>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            label = html.escape(it.get("label", ""))
            value = html.escape(it.get("value", ""))
            comment = html.escape(it.get("comment", ""))
            st.markdown(
                f"""
<div class="ui-kpi">
  <div class="ui-kpi-label">{label}</div>
  <div class="ui-kpi-value">{value}</div>
  <div class="ui-kpi-comment">{comment}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def render_table_report(
    df: pd.DataFrame,
    *,
    formats: dict[str, str] | None = None,
    numeric_cols: list[str] | None = None,
) -> None:
    _df = df.copy()

    if formats:
        for col, fmt in formats.items():
            if col in _df.columns:
                try:
                    _df[col] = _df[col].map(lambda x: fmt.format(x) if pd.notna(x) else "")
                except Exception:
                    pass

    if numeric_cols is None:
        numeric_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(df[c])]

    cols = list(_df.columns)
    thead = "<thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead>"

    body_rows = []
    for _, row in _df.iterrows():
        tds = []
        for c in cols:
            v = row[c]
            v_str = "" if pd.isna(v) else str(v)
            cls = "num" if c in numeric_cols else ""
            tds.append(f"<td class='{cls}'>{html.escape(v_str)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    table = f"<table class='ui-report'>{thead}{tbody}</table>"

    st.markdown(
        f"""
<div class="ui-report-wrap">
  <div class="ui-report-card">{table}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_style(fig: Any) -> None:
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=UI["primary"],
            font=dict(color=UI["text"], size=14),
            align="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12, color=UI["muted"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=UI["text"]),
        xaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False),
        margin=dict(l=10, r=10, t=60, b=10),
    )


def term(label: str, definition: str, *, key: str | None = None) -> None:
    safe_label = html.escape(label)
    safe_def = html.escape(definition)
    c1, c2 = st.columns([0.001, 1], vertical_alignment="center")
    with c1:
        st.markdown(
            f"<span title='{safe_def}' style='font-weight:800;color:{UI['text']};'>{safe_label}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        with st.popover("ⓘ", use_container_width=False):
            st.markdown(f"**{label}**")
            st.write(definition)


def row_start(key: str) -> None:
    safe_key = html.escape(key)
    st.markdown(f"<div id='{safe_key}'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
<style>
div:has(#{safe_key}) {{
  background: {UI["panel"]};
  border: 1px solid {UI["border"]};
  border-radius: 16px;
  box-shadow: 0 14px 26px {UI["shadow"]};
  padding: 10px 12px;
  margin: 10px 0;
}}
</style>
        """,
        unsafe_allow_html=True,
    )