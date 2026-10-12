"""Reactive baseline versus scenario comparison using the same fitted model."""
import pandas as pd
import streamlit as st
from modules.config import pretty
from modules.prediction import predict_single, risk_category, explain_single, sensitivity_explanation
from views.common import page_header, table


def show_explanation(ctx, values):
    local = explain_single(ctx.trained, ctx.model_name, values)
    if local is not None:
        st.subheader("Academic indicators associated with this estimate")
        table(local)
        st.caption("Exact Logistic Regression contributions in log-odds relative to the model intercept: coefficient times standardized, imputed value. Positive contributions raise the score. Associations do not establish causes.")
    else:
        st.subheader("Prediction sensitivity to academic indicators")
        table(sensitivity_explanation(ctx.trained, ctx.model_name, values))
        st.caption("For each indicator, replace only that value with its training-set median. Score difference = current estimate minus the replacement estimate. These differences are not additive contributions or causal effects. Correlated indicators can distort interpretation.")
    with st.expander("Global model explanation"):
        table(ctx.trained["results"][ctx.model_name]["importance"])
        st.caption("Global coefficients / tree importance / training-set permutation importance describe the fitted model, not the cause of any student's outcome.")


def render(ctx):
    page_header("What-If Simulator", "Compare academic scenarios using the same trained model.")
    st.warning("Simulations are estimates, not guaranteed outcomes. Scores are not calibrated chances of failing. Changing an input does not prove that an intervention will cause this result.")
    if not ctx.has_model:
        st.info(ctx.training_error)
        return
    if ctx.view.empty:
        st.info("No records match the selected filters.")
        return
    st.markdown("**1. Choose a baseline record**")
    index = st.selectbox("Baseline student record", ctx.view.index.tolist(), format_func=lambda i: str(ctx.view.loc[i, "student_id"]) + (f" / semester {ctx.view.loc[i, 'semester']}" if "semester" in ctx.view else f" / record {i + 1}"))
    features = ctx.trained["features"]
    baseline = ctx.cleaned.loc[index, features].to_dict()
    values = baseline.copy()
    st.caption(f"Model: {ctx.model_name}. Baseline and scenario both use this fitted model; baseline can differ from the out-of-fold score in Student Records. Missing baseline values use training medians; entirely empty training features use a disclosed zero fallback.")
    medians = ctx.trained["models"][ctx.model_name].named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_
    dataset_key = int(pd.util.hash_pandas_object(ctx.cleaned[features], index=True).sum())
    st.markdown("**2. Adjust the academic indicators**")
    st.caption("Move a slider to update the comparison immediately. All scores use a 0-100 scale.")
    cols = st.columns(2)
    for i, feature in enumerate(features):
        value = baseline[feature]
        default = float(medians[i] if pd.isna(value) else value)
        values[feature] = cols[i % 2].slider(pretty(feature), 0.0, 100.0, default, 1.0, key=f"scenario_{index}_{feature}_{dataset_key}")
    before = predict_single(ctx.trained, ctx.model_name, baseline)
    after = predict_single(ctx.trained, ctx.model_name, values)
    st.markdown("**3. Compare the estimates**")
    a, b = st.columns(2)
    a.metric("Baseline estimate", f"{before:.1%}")
    a.write(risk_category(before, ctx.moderate, ctx.high))
    b.metric("Simulated estimate", f"{after:.1%}", f"{(after-before)*100:+.1f} percentage points", delta_color="inverse")
    b.write(risk_category(after, ctx.moderate, ctx.high))
    with st.expander("See the exact input changes"):
        table(pd.DataFrame({"Indicator": [pretty(f) for f in features], "Baseline": [baseline[f] for f in features], "Simulated": [values[f] for f in features]}))
    show_explanation(ctx, values)
