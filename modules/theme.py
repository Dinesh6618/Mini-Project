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
[data-testid="stHeader"] {{ background: rgba(11,18,32,.92); }}
[data-testid="stSidebar"] {{ background: #0f1929; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stSidebar"] h1 {{ font-size: 1.35rem !important; letter-spacing: -.03em; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: .12rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .35rem .6rem; border-radius: 8px; transition: background .15s; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #192a40; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: #163c40; box-shadow: inset 3px 0 {TEAL}; }}
[data-testid="stMetric"] {{ background: linear-gradient(135deg, #152338, {SURFACE}); border: 1px solid {BORDER}; padding: 19px 20px; border-radius: 14px; min-height: 118px; }}
[data-testid="stMetricLabel"] p {{ color: {INK_MUTED}; font-size: .85rem; }}
[data-testid="stMetricValue"] {{ font-size: clamp(1.5rem, 2.3vw, 2rem); font-weight: 650; letter-spacing: -.04em; }}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ border-radius: 14px !important; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}
h1 {{ font-size: clamp(1.8rem, 3vw, 2.5rem) !important; letter-spacing: -.04em; line-height: 1.2 !important; padding-bottom: .5rem !important; }}
h2 {{ font-size: 1.4rem !important; letter-spacing: -.02em; }}
h3 {{ font-size: 1.05rem !important; letter-spacing: -.01em; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }}
.page-caption {{ color: {INK_MUTED}; line-height: 1.65; max-width: 850px; margin: 0 0 1.25rem; }}
.eyebrow {{ color: #79d9cb; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .4rem; }}
.context-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; padding: 0 0 1.3rem; }}
.context-pill {{ font-size: .76rem; color: {INK_MUTED}; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 99px; padding: .35rem .75rem; }}
.context-pill.accent {{ color: #8de3d5; background: #103631; border-color: #24584e; }}
.insight-panel {{ padding: 20px 24px; border-radius: 14px; border: 1px solid #2b4e55; background: linear-gradient(115deg, #10332f, #132238); margin: 1.25rem 0; }}
.insight-panel h3 {{ margin: 0 0 .35rem; color: #c8f6ed; }}
.insight-panel p {{ margin: 0; color: #bbd2d8; line-height: 1.65; }}
.chart-note {{ color: {INK_MUTED}; font-size: .82rem; line-height: 1.6; margin: .3rem 0; }}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{ min-height: 42px; border-radius: 9px; font-weight: 600; }}
button:focus-visible, a:focus-visible {{ outline: 3px solid #79d9cb !important; outline-offset: 3px; }}
[data-testid="stExpander"] {{ border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
[data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {BORDER}; margin: 1rem 0; }}
[data-baseweb="tab"] {{ padding: .75rem .15rem; font-weight: 600; }}
.risk-badge {{ display: inline-block; padding: 6px 14px; border-radius: 6px; font-weight: 600; border: 1px solid {BORDER}; background: {SURFACE}; }}
@media (max-width: 700px) {{
 .block-container {{ padding: 3.8rem 1rem 2rem; }}
 [data-testid="stMetric"] {{ min-height: 100px; padding: 14px; }}
 .insight-panel {{ padding: 16px; }}
 [data-baseweb="tab-list"] {{ gap: .7rem; }}
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
