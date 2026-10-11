"""Exploratory Analysis page."""

from __future__ import annotations

import streamlit as st

from modules import exploratory_analysis as eda
from modules.config import subject_columns
from views.common import chart, empty_view, page_header, table


def render(ctx) -> None:
    page_header(
        "Exploratory Analysis",
        "Charts describe patterns in the records currently selected by the sidebar filters.",
    )
    if empty_view(ctx):
        return
    view = ctx.view

    tabs = st.tabs([
        "Distributions", "Relationships", "Assignments", "Progress", "Subjects", "Outliers",
    ])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            chart(
                eda.histogram(view, "internal_marks"),
                "Spread of internal marks. Colours show the historical result where known.",
            )
        with right:
            chart(
                eda.histogram(view, "attendance_percentage"),
                "Spread of attendance percentages across students.",
            )
        stats = eda.descriptive_statistics(view)
        if stats is not None:
            st.subheader("Summary statistics")
            st.dataframe(stats, width="stretch")

    with tabs[1]:
        st.info(
            "Correlation measures how two numbers move together in this dataset. "
            "It does NOT show that one causes the other: other factors (prior preparation, "
            "workload, health, teaching) may influence both."
        )
        result = eda.scatter_with_trend(
            view, "attendance_percentage", "internal_marks",
            "Attendance versus internal marks", hover=["student_id"],
        )
        if result is not None:
            fig, r, n = result
            chart(fig, f"{n} students. Pearson r = {r:.2f} ({eda.describe_correlation(r)})." if r is not None else "")
        else:
            chart(None)
        correlations = eda.attendance_correlations(view)
        if correlations is not None:
            st.subheader("Attendance and academic scores")
            table(correlations)
        matrix = eda.correlation_matrix(view)
        if matrix is not None:
            chart(
                eda.correlation_heatmap(matrix),
                "Teal = move together, red = move in opposite directions, grey = little relationship.",
            )

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            chart(eda.histogram(view, "assignment_score"), "Distribution of assignment scores.")
        with right:
            chart(
                eda.assignment_by_outcome(view),
                "Assignment scores for students with each historical result.",
                empty="Needs assignment_score and a final_result column.",
            )
        result = eda.scatter_with_trend(
            view, "assignment_score", "internal_marks", "Assignment score versus internal marks",
        )
        if result is not None:
            fig, r, n = result
            chart(fig, f"{n} students. Pearson r = {r:.2f} ({eda.describe_correlation(r)})." if r is not None else "")

    with tabs[3]:
        result = eda.previous_vs_current(view)
        left, right = st.columns(2)
        with left:
            if result is None:
                chart(None, empty="Needs previous_semester_marks and internal_marks.")
            else:
                fig, r, n = result
                chart(
                    fig,
                    "Points below the dotted line scored lower than in the previous semester; "
                    f"points above scored higher ({n} students).",
                )
        with right:
            chart(eda.change_histogram(view), "Most students cluster near zero; the left tail shows declines.")
        change = eda.change_in_marks(view)
        if change is not None and len(change):
            declined = int((change < -ctx.indicators["decline_points"]).sum())
            st.caption(
                f"{declined} of {len(change)} students ({declined / len(change):.0%}) dropped by more than "
                f"{ctx.indicators['decline_points']:.0f} points. This is a prompt for a supportive "
                "conversation, not a conclusion."
            )

    with tabs[4]:
        if not subject_columns(view):
            st.info("No subject-mark columns (for example statistics_marks) were found in this dataset.")
        else:
            left, right = st.columns(2)
            with left:
                chart(eda.subject_bar_chart(view), "Average marks for each subject.")
            with right:
                chart(eda.subject_box_chart(view), "Boxes show the middle 50% of marks; dots are unusual values.")
            table(eda.subject_summary(view).round(1))

    with tabs[5]:
        st.caption(
            "Values outside 1.5 x the interquartile range (IQR) of a measure are listed for human "
            "review. An outlier is not an error and is never a reason to exclude a student."
        )
        outliers = eda.find_outliers(view)
        if outliers.empty:
            st.success("No statistical outliers were found in the selected records.")
        else:
            counts = outliers.groupby("Measure").size().rename("Flagged values").reset_index()
            left, right = st.columns([1, 2])
            with left:
                table(counts)
            with right:
                table(outliers)
