"""Small helpers shared by all pages."""

from __future__ import annotations

import html
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def page_header(title: str, caption: str = "") -> None:
    st.markdown('<div class="eyebrow">Student success / Academic insights</div>', unsafe_allow_html=True)
    st.title(title)
    if caption:
        st.markdown(f'<p class="page-caption">{caption}</p>', unsafe_allow_html=True)


def chart(fig: go.Figure | None, note: str = "", empty: str = "This chart needs columns that are not in the dataset.") -> None:
    """Draw a Plotly figure with an optional explanation underneath."""
    if fig is None:
        st.info(empty)
        return
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", theme=None)
        if note:
            st.markdown(f'<p class="chart-note">{html.escape(note)}</p>', unsafe_allow_html=True)


ROW_HEIGHT = 35  # pixels per table row in Streamlit's dataframe grid


def table(frame: pd.DataFrame | None, max_height: int = 560, **kwargs) -> None:
    """Show a table sized to its content (scrolls only when it is long)."""
    if frame is None or frame.empty:
        st.caption("Nothing to show.")
        return
    height = min(ROW_HEIGHT * (len(frame) + 1) + 3, max_height)
    st.dataframe(frame, width="stretch", height=height, hide_index=True, **kwargs)


def empty_view(ctx) -> bool:
    """Show an empty state and return True when the filtered view has no rows."""
    if len(ctx.view) == 0:
        st.warning(
            "No students match the current filters. Clear a filter in the sidebar to see data."
        )
        return True
    return False


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def navigate(page: str) -> None:
    """Button callback: switch the existing navigation on the next rerun."""
    st.session_state["page"] = page


def context_bar(ctx) -> None:
    """Compact, factual dataset status shown on every page."""
    labels = [
        ("Synthetic demo data" if ctx.using_sample else "Your dataset", "accent"),
        (f"{len(ctx.view):,} of {len(ctx.cleaned):,} records in view", ""),
        ("Anonymous IDs on" if ctx.anonymize else "Identifiable records", ""),
        ("Model ready" if ctx.has_model else "Descriptive analysis only", ""),
    ]
    pills = "".join(f'<span class="context-pill {kind}">{html.escape(label)}</span>' for label, kind in labels)
    st.markdown(f'<div class="context-bar">{pills}</div>', unsafe_allow_html=True)
