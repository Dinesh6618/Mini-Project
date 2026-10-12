"""Academic group comparisons supported by the current data."""
import streamlit as st
import plotly.express as px
from modules.config import subject_columns
from modules.advanced import class_comparison
from views.common import page_header, table


def render(ctx):
    page_header("Class and Subject Comparison", "Compare observed means and contributing record counts.")
    data = ctx.view
    groups = [c for c in ["department", "year", "section"] if c in data and data[c].notna().any()]
    subjects = [c for c in subject_columns(data) if data[c].notna().any()]
    if data.empty:
        st.info("No records match the selected filters.")
        return
    if not groups:
        st.info("Upload department, year or section to compare classes.")
        return
    group = st.selectbox("Compare by", groups)
    options = list(dict.fromkeys(subjects + [c for c in ["internal_marks", "attendance_percentage", "assignment_score"] if c in data and data[c].notna().any()]))
    if not subjects:
        st.info("No subject columns are available. Upload subject columns ending in _marks for subject comparisons; core indicators remain available.")
    chosen = st.multiselect("Subjects and indicators", options, default=options)
    means, counts = class_comparison(data, group, chosen)
    if means.empty:
        st.info("Choose at least one indicator with available group records.")
        return
    st.plotly_chart(px.imshow(means, text_auto=".1f", zmin=0, zmax=100, color_continuous_scale="Teal", aspect="auto", labels={"color": "Mean / 100"}), width="stretch")
    st.subheader("Group means")
    table(means.reset_index())
    st.subheader("Nonmissing records per mean")
    table(counts.reset_index())
    st.caption("Means exclude missing values and use the sidebar filters. Each semester record contributes once; repeated students can contribute several records. Group differences are descriptive associations, not rankings or causal effects.")
