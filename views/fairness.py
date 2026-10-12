"""Privacy-preserving academic group error review."""
import streamlit as st
import plotly.express as px
from modules.advanced import fairness_audit
from views.common import page_header, table


def render(ctx):
    page_header("Bias and Fairness Auditor", "Review errors on the same held-out records used for model evaluation.")
    st.warning("Differences in errors do not prove bias or fairness. Small samples, historical labels and repeated semester records limit interpretation. Do not use this audit to penalize students or groups.")
    if not ctx.has_model:
        st.info(ctx.training_error)
        return
    groups = [c for c in ["department", "year", "section"] if c in ctx.cleaned]
    if not groups:
        st.info("No appropriate academic grouping columns are available.")
        return
    st.caption("Only academic groups are offered; sensitive attributes are never model inputs or selectable audit fields. Groups with fewer than 10 distinct held-out students are suppressed, including their names. Each error-rate denominator needs 10 distinct students of that outcome. Missing attributes are excluded. This threshold is a disclosure safeguard, not a statistical adequacy guarantee.")
    group = st.selectbox("Audit attribute", groups)
    name = st.selectbox("Audit model", list(ctx.trained["models"]))
    audit = fairness_audit(ctx.cleaned, ctx.trained, name, group)
    st.info("This audit uses the full held-out split and ignores sidebar filters, which could otherwise create small identifying groups. Predictions use the classifier decision rule, not sidebar risk category thresholds.")
    if len(audit) < 2:
        st.info("At least two groups meeting the privacy threshold are required for a comparison. More independently labelled students may be needed.")
        return
    table(audit)
    long = audit.melt(id_vars="Group", value_vars=["Error rate", "False positive rate", "False negative rate"], var_name="Metric", value_name="Rate").dropna()
    st.plotly_chart(px.bar(long, x="Group", y="Rate", color="Metric", barmode="group"), width="stretch")
    st.caption("False positive: flagged non-pass among actual passes. False negative: missed among actual non-passes. Missing rates are suppressed due to insufficient outcome-specific students. Metrics weight semester records, not individual students; no confidence or significance claim is made.")
