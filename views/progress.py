"""Semester history without inventing missing periods."""
import streamlit as st
import pandas as pd
import plotly.express as px
from modules.advanced import semester_progress
from modules.config import CORE_FEATURES
from views.common import page_header, table


def render(ctx):
    page_header("Semester Progress Tracker", "Observed changes across uploaded semester records.")
    try:
        data = semester_progress(ctx.view)
    except ValueError as exc:
        st.info(str(exc))
        return
    if data.empty:
        st.info("No records match the filters.")
        return
    student = st.selectbox("Student", data.student_id.unique().tolist())
    history = data[data.student_id == student]
    st.caption("Uses records in the current filters. Changes are from the preceding available semester, which may skip periods. Missing values are not filled.")
    if len(history) < 2:
        st.info("At least two semesters for this student are needed to identify change.")
    columns = st.multiselect("Academic indicators", CORE_FEATURES, default=CORE_FEATURES[:2])
    if columns:
        long = history.melt(id_vars="semester", value_vars=columns, var_name="Indicator", value_name="Value")
        st.plotly_chart(px.line(long, x="semester", y="Value", color="Indicator", markers=True), width="stretch")
        for col in columns:
            change = history.iloc[-1][col + "_change"]
            if pd.notna(change):
                st.write(f"{col.replace('_', ' ').title()}: {'improved' if change > 0 else 'declined' if change < 0 else 'unchanged'} ({change:+.1f} points since preceding available semester).")
    table(history)
