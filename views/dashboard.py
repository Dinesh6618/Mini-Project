"""Dashboard page: headline metrics and overview charts."""

from __future__ import annotations

import html
import pandas as pd
import plotly.express as px
import streamlit as st

from modules import exploratory_analysis as eda
from modules.config import RISK_HIGH, RISK_LOW, RISK_MODERATE, TARGET_COLUMN
from modules.recommendations import needs_follow_up, recommendation_summary
from modules.theme import RISK_COLORS, style_figure
from views.common import chart, empty_view, page_header, table, navigate


def _support_metric(ctx) -> tuple[str, int, str]:
    view = ctx.view
    if ctx.has_risk:
        count = int(view["risk_category"].isin([RISK_MODERATE, RISK_HIGH]).sum())
        return (
            "May need support",
            count,
            f"Students with Moderate or High estimated risk ({ctx.model_name}). "
            "Thresholds can be changed in the sidebar. These are estimates, not decisions.",
        )
    count = int(needs_follow_up(view, ctx.indicators).sum())
    return (
        "Below an indicator",
        count,
        "No trained model is available, so this counts students below at least one indicator "
        f"threshold (attendance < {ctx.indicators['attendance_percentage']:.0f}%, internal marks "
        f"< {ctx.indicators['internal_marks']:.0f}, assignment < {ctx.indicators['assignment_score']:.0f}, "
        f"or a drop of more than {ctx.indicators['decline_points']:.0f} points).",
    )


def render(ctx) -> None:
    page_header(
        "Academic Performance Dashboard",
        f"{ctx.source_label} &middot; a decision-support view. Figures describe groups of "
        "records and are not judgements about individual students.",
    )
    if empty_view(ctx):
        return
    view = ctx.view

    label, count, help_text = _support_metric(ctx)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", f"{len(view):,}")
    internal = view["internal_marks"].mean() if "internal_marks" in view else float("nan")
    attendance = view["attendance_percentage"].mean() if "attendance_percentage" in view else float("nan")
    c2.metric("Average internal marks", "N/A" if pd.isna(internal) else f"{internal:.1f} / 100")
    c3.metric("Average attendance", "N/A" if pd.isna(attendance) else f"{attendance:.1f}%")
    c4.metric(label, f"{count:,}", help=help_text)

    if ctx.has_risk:
        headline = f"{count:,} records may benefit from a closer look"
        detail = f"These records have moderate or high estimated risk using {ctx.model_name}. Start with the academic indicators and use individual context before planning support."
    else:
        headline = "Start with the academic indicators"
        detail = f"{count:,} records fall below at least one selected academic threshold. Historical outcomes are needed before model estimates can be shown."
    st.markdown(f'<div class="insight-panel"><h3>{html.escape(headline)}</h3><p>{html.escape(detail)}</p></div>', unsafe_allow_html=True)
    st.subheader("What would you like to do?")
    actions = [
        ("Review student records", "Find a student and review their academic indicators.", "Open student records", "Student Records"),
        ("Explore a possible change", "Move academic sliders and compare model estimates.", "Try the simulator", "What-If Simulator"),
        ("Share your analysis", "Export a PDF summary or the filtered CSV dataset.", "Open report generator", "Report Generator"),
    ]
    for col, (title, description, label, destination) in zip(st.columns(3), actions):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
                st.button(label, key=f"go_{destination}", on_click=navigate, args=(destination,), width="stretch")
    with st.expander("New here? A simple walkthrough"):
        st.markdown("1. **Choose data** in the sidebar, or explore the clearly labelled synthetic sample.\n2. **Read the overview** and filter the records you want to examine.\n3. **Review students or try a scenario** to understand the academic indicators.\n4. **Check model evaluation and data quality**, then export a report.")
        st.caption("Risk estimates suggest where to offer support. They are not guaranteed outcomes or decisions about a student.")

    if ctx.filters_active:
        st.caption(f"Filters are active: showing {len(view):,} of {len(ctx.scored):,} students.")

    overview_tab, trends_tab, support_tab = st.tabs(["Performance overview", "Subjects & progress", "Support indicators"])
    with overview_tab:
        left, right = st.columns(2)
        with left:
            chart(
                eda.performance_band_chart(view),
                "How many students fall into each internal-marks band.",
            )
        with right:
            result = eda.scatter_with_trend(
                view, "attendance_percentage", "internal_marks",
                "Attendance versus internal marks", hover=["student_id", "department"],
            )
            if result is None:
                chart(None)
            else:
                fig, r, n = result
                note = f"Each point is one student ({n} shown)."
                if r is not None:
                    note += (
                        f" Pearson r = {r:.2f} ({eda.describe_correlation(r)}). "
                        "This is an association, not proof that attendance causes higher marks."
                    )
                chart(fig, note)

    with trends_tab:
        left, right = st.columns(2)
        with left:
            chart(
                eda.subject_bar_chart(view),
                "Average marks per subject column found in the data.",
                empty="No subject-mark columns (for example statistics_marks) were found.",
            )
        with right:
            group = next(
                (g for g in ("year", "department", "section")
                 if g in view.columns and view[g].nunique() > 1),
                None,
            )
            trend = eda.performance_trend_by_group(view, group) if group else None
            if trend is not None:
                chart(trend, f"Average previous-semester marks compared with current internal marks, by {group}.")
            else:
                chart(
                    eda.change_histogram(view),
                    "Change in marks since the previous semester (right of the line = improvement).",
                    empty="Trends need previous_semester_marks and internal_marks.",
                )

    with support_tab:
        if ctx.has_risk:
            left, right = st.columns(2)
            counts = (
                view["risk_category"].value_counts()
                .reindex([RISK_LOW, RISK_MODERATE, RISK_HIGH, "Not scored"], fill_value=0)
                .rename_axis("Risk category").reset_index(name="Students")
            )
            counts = counts[counts["Students"] > 0]
            fig = px.bar(
                counts, x="Risk category", y="Students", color="Risk category",
                color_discrete_map=RISK_COLORS, text="Students",
            )
            fig.update_layout(showlegend=False)
            fig.update_traces(textposition="outside", cliponaxis=False)
            with left:
                chart(
                    style_figure(fig, f"Estimated risk category ({ctx.model_name})"),
                    f"Moderate starts at {ctx.moderate:.0%} and High at {ctx.high:.0%} estimated "
                    "probability of a non-pass outcome.",
                )
            with right:
                st.subheader("Indicators below threshold")
                summary = recommendation_summary(view, ctx.indicators)
                table(
                    summary, column_config={
                        "Share of students": st.column_config.ProgressColumn(
                            "Share of students", format="percent", min_value=0.0, max_value=1.0,
                        )
                    },
                )
                st.caption("Thresholds are adjustable in the sidebar under Risk settings.")
        else:
            st.subheader("Indicators below threshold")
            table(recommendation_summary(view, ctx.indicators))
            if ctx.training_error:
                st.info(
                    "Risk estimates are unavailable: " + ctx.training_error
                    + " The indicator counts above use simple thresholds only."
                )

    if TARGET_COLUMN in view.columns:
        known = view[TARGET_COLUMN].dropna()
        if len(known):
            st.caption(
                f"Historical outcomes in view: {int((known == 'Pass').sum())} Pass, "
                f"{int((known == 'Fail').sum())} Fail, {int(view[TARGET_COLUMN].isna().sum())} unknown."
            )
