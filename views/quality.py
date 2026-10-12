"""Transparent data-quality score."""
import streamlit as st
from modules.advanced import quality_assessment
from views.common import page_header, table


def render(ctx):
    page_header("Data Quality Score", "An explicit quality checklist for the original loaded dataset, before cleaning.")
    score, components = quality_assessment(ctx.raw)
    st.metric("Dataset quality score", f"{score:.1f} / 100")
    st.markdown("**Calculation:** 100 * (1 - mean of the four issue rates below). Each check has equal weight (25%). This is a configurable policy choice represented here by a fixed, disclosed checklist, not a validated measure of predictive reliability.")
    table(components)
    st.caption("Missing cells include blank strings in all columns. Invalid values count non-numeric / out-of-range academic indicators and unrecognized nonblank outcome labels. Duplicate rate counts extra exact rows. Conflict rate counts all distinct rows sharing a student and semester (or student alone without semester). Checks can overlap; optional missing columns are not penalized. Sidebar filters do not change this source-data assessment.")
    st.subheader("Detected issues and cleaning actions")
    table(ctx.report["issues"])
    for warning in ctx.report["warnings"]:
        st.warning(warning)
    st.info("Detailed row-level changes are available on Data Cleaning. No missing academic values are filled for analysis.")
