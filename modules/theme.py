"""Colours, page styling and a shared Plotly look.

The categorical palette was checked with the data-viz palette validator against
the card surface (#111c2d): lightness band, chroma, colour-blind separation and
3:1 contrast all pass for four series. Teal is the primary colour; amber,
violet and coral follow in fixed order.
"""

from __future__ import annotations

import plotly.graph_objects as go

NAVY = "#0b1220"        # page background
SURFACE = "#111c2d"     # cards, sidebar, chart surface
BORDER = "#26364a"
INK = "#e6edf5"         # primary text
INK_MUTED = "#a9bacd"   # secondary text
GRID = "#223247"

TEAL = "#1fa595"
AMBER = "#d07d27"
VIOLET = "#7683ea"
CORAL = "#d9534f"
CATEGORICAL = [TEAL, AMBER, VIOLET, CORAL]

# Outcome and risk colours are always shown together with a text label.
OUTCOME_COLORS = {"Pass": TEAL, "Fail": CORAL, "Unknown": INK_MUTED}
RISK_COLORS = {
    "Low Risk": TEAL,
    "Moderate Risk": AMBER,
    "High Risk": CORAL,
    "Not scored": INK_MUTED,
}

# One-hue sequential ramp (light -> dark would vanish on navy, so dark -> teal).
SEQUENTIAL = ["#16384a", "#17636b", "#1fa595", "#6fd3c4"]
# Diverging: coral <-> neutral <-> teal, for correlations.
DIVERGING = [[0.0, CORAL], [0.5, "#2a3a50"], [1.0, TEAL]]

CSS = f"""
<style>
.stApp {{ background: {NAVY}; color: {INK}; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER};
    padding: 12px 16px; border-radius: 8px;
}}
[data-testid="stMetricLabel"] p {{ color: {INK_MUTED}; font-size: 0.85rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
div[data-testid="stAlert"] {{ border-radius: 8px; }}
h1 {{ font-size: 1.7rem; margin-bottom: 0.2rem; }}
h2, h3 {{ color: {INK}; }}
.block-container {{ padding-top: 2rem; max-width: 1250px; }}
.page-caption {{ color: {INK_MUTED}; margin: 0 0 1rem 0; }}
.chart-note {{ color: {INK_MUTED}; font-size: 0.85rem; margin: -0.4rem 0 1rem 0; }}
.risk-badge {{
    display: inline-block; padding: 6px 14px; border-radius: 6px;
    font-weight: 600; border: 1px solid {BORDER}; background: {SURFACE};
}}
</style>
"""


def style_figure(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the shared dark, low-ink look to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0.0, xanchor="left",
                   y=0.97, yanchor="top")
        if title else None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_MUTED, size=12),
        margin=dict(l=55, r=20, t=80 if title else 30, b=50),
        height=height,
        # Legend sits on its own row under the title so the two never collide.
        legend=dict(
            title_text="", bgcolor="rgba(0,0,0,0)", font=dict(color=INK_MUTED),
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
        ),
        colorway=CATEGORICAL,
        hoverlabel=dict(bgcolor=NAVY, font_color=INK, bordercolor=BORDER),
        bargap=0.15,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, title_font=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, title_font=dict(color=INK_MUTED))
    return fig
