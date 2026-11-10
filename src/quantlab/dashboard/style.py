"""Visual tokens and plotly helpers for the dashboard.

Colors follow the validated reference palette (categorical slots in fixed order, a one-hue
sequential ramp, a blue<->red diverging pair with a gray midpoint). Home team = slot 1 (blue),
away team = slot 2 (orange), a third series (e.g. "actual") = slot 3 (aqua). Light and dark
steps are selected separately; text never wears a series color.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
         "gray": "#c3c2b7", "neg": "#e34948", "pos": "#2a78d6", "mid": "#f0efec", "seq_light": "#cde2fb"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500",
        "gray": "#52514e", "neg": "#e66767", "pos": "#3987e5", "mid": "#383835", "seq_light": "#184f95"}


def tokens() -> dict:
    try:
        kind = st.context.theme.type
    except Exception:
        kind = "light"
    return DARK if kind == "dark" else LIGHT


def team_colors() -> tuple[str, str]:
    t = tokens()
    return t["s1"], t["s2"]


def base_layout(fig: go.Figure, height: int = 320, title: str | None = None, xtitle: str | None = None,
                ytitle: str | None = None, legend: bool = True) -> go.Figure:
    t = tokens()
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=40 if title else 12, b=8),
        title=dict(text=title, font=dict(size=14, color=t["ink"]), x=0, xanchor="left") if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=t["ink2"], size=12),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                                       font=dict(color=t["ink2"])),
        hoverlabel=dict(font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif')),
        bargap=0.12,
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], ticks="", title=xtitle, zeroline=False,
                     tickfont=dict(color=t["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=t["grid"], gridwidth=1, linecolor=t["axis"], title=ytitle,
                     zeroline=False, tickfont=dict(color=t["muted"]))
    return fig


def css() -> None:
    st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.ql-note { color: var(--text-color); opacity: .72; font-size: .86rem; }
.ql-chip { display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .78rem;
           border: 1px solid rgba(127,127,127,.35); margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


def pct(x: float) -> str:
    return "–" if x != x else f"{100 * x:.0f}%"


def odds(x: float) -> str:
    return "–" if x != x else (f"{x:+.0f}")
