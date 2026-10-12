"""Risk Prediction page: single-student estimate and transparent thresholds."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.config import RISK_HIGH, RISK_LOW, RISK_MODERATE, pretty
from modules.prediction import explain_single, predict_single, risk_category
from modules.recommendations import academic_recommendations, indicator_flags
from modules.theme import INK_MUTED, RISK_COLORS, TEAL, style_figure
from views.common import chart, page_header, table


def _no_model_state(ctx) -> None:
    st.warning(ctx.training_error or "A predictive model could not be trained.")
    st.markdown(
        "**Why there is no prediction here:** a risk model has to learn from historical "
        "outcomes. Without them, any 'prediction' would be invented.\n\n"
        "**What the dataset needs:**\n"
        "- a `final_result` column with `Pass` / `Fail` for past students\n"
        "- at least 40 labelled records, with at least 8 examples of each outcome\n"
        "- academic indicators recorded *before* the result (attendance, internal marks, "
        "assignment scores, previous marks)"
    )
    st.subheader("What you can still do: indicator review")
    flags = indicator_flags(ctx.view, ctx.indicators)
    if flags.empty:
        st.info("No indicator columns are available.")
        return
    summary = pd.DataFrame({
        "Indicator": flags.columns,
        "Students": [int(flags[c].sum()) for c in flags.columns],
    })
    table(summary)
    st.caption("These counts use the threshold values in the sidebar. They are not predictions.")


def render(ctx) -> None:
    page_header(
        "Risk Prediction",
        "Estimate which students may benefit from extra academic support, using models "
        "trained on historical outcomes.",
    )
    st.warning(
        "These categories are estimates from past patterns, not guaranteed outcomes and not "
        "decisions about a student. Use them only to start a supportive conversation."
    )
    if not ctx.has_model:
        _no_model_state(ctx)
        return

    trained = ctx.trained
    model_name = ctx.model_name
    features = trained["features"]

    st.markdown(
        f"**Model:** {model_name} &nbsp;|&nbsp; **Target:** historical non-pass (`final_result = Fail`) "
        f"&nbsp;|&nbsp; **Trained on:** {trained['n_train']} records "
        f"(held-out test: {trained['n_test']})"
    )
    st.markdown(
        f"**Category thresholds (adjust in the sidebar):** Low Risk below {ctx.moderate:.0%}; "
        f"Moderate Risk from {ctx.moderate:.0%} to below {ctx.high:.0%}; "
        f"High Risk at {ctx.high:.0%} or above."
    )
    st.caption(
        "Logistic Regression and tree models give extra weight to the less common non-pass outcome; KNN uses unweighted neighbours. "
        "Scores are relative model estimates, not calibrated chances of failing. Associations do not establish causes. Entirely empty training features use zero imputation; other missing values use training medians."
    )

    st.subheader("Estimate for one student")
    st.caption("Enter indicators that would be known before the final result.")
    with st.form("single_student"):
        cols = st.columns(2)
        values: dict[str, float] = {}
        for i, feature in enumerate(features):
            series = ctx.cleaned[feature].dropna() if feature in ctx.cleaned else pd.Series(dtype=float)
            default = float(series.median()) if len(series) else 50.0
            label = pretty(feature)
            if feature != "attendance_percentage":
                label += " (0-100)"
            values[feature] = cols[i % 2].number_input(
                label, min_value=0.0, max_value=100.0,
                value=float(min(100.0, max(0.0, round(default)))), step=1.0, format="%.0f",
                key=f"input_{feature}",
            )
        submitted = st.form_submit_button("Estimate risk", type="primary")

    if submitted:
        probability = predict_single(trained, model_name, values)
        category = risk_category(probability, ctx.moderate, ctx.high)
        m1, m2 = st.columns([1, 2])
        m1.metric("Estimated category", category)
        m2.metric("Estimated non-pass probability", f"{probability:.1%}")
        if category == RISK_LOW:
            st.success("Low estimated risk. Regular support applies.")
        elif category == RISK_MODERATE:
            st.warning("Moderate estimated risk. A friendly check-in may be useful.")
        else:
            st.error("High estimated risk. Consider offering additional academic support.")

        from views.simulator import show_explanation
        show_explanation(ctx, values)

        tips = academic_recommendations(values, ctx.indicators)
        st.markdown("**Supportive suggestions**")
        if tips:
            for tip in tips:
                st.write(f"- {tip}")
        else:
            st.write(
                "None of the simple indicators is below its threshold. Individual context and "
                "professional judgement still matter."
            )

    st.divider()
    if ctx.has_risk and "risk_probability" in ctx.view.columns:
        scored = ctx.view.dropna(subset=["risk_probability"])
        if len(scored):
            fig = px.histogram(
                scored, x="risk_probability", nbins=20, color_discrete_sequence=[TEAL],
                labels={"risk_probability": "Estimated non-pass probability"},
            )
            fig.add_vline(x=ctx.moderate, line_dash="dash", line_color=RISK_COLORS[RISK_MODERATE],
                          annotation_text="Moderate", annotation_font_color=INK_MUTED)
            fig.add_vline(x=ctx.high, line_dash="dash", line_color=RISK_COLORS[RISK_HIGH],
                          annotation_text="High", annotation_font_color=INK_MUTED)
            fig.update_yaxes(title="Number of students")
            chart(
                style_figure(fig, "Where the thresholds fall"),
                "Distribution of estimates for the students in view. Students who already have "
                "a known result are scored out-of-fold (by a model that did not train on them).",
            )
