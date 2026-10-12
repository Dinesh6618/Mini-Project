"""Model Evaluation page: held-out test metrics and interpretation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import model_evaluation as ev
from views.common import chart, page_header, table


def render(ctx) -> None:
    page_header(
        "Model Comparison Lab",
        "Every number on this page is calculated from a held-out test set that the models "
        "did not see during training.",
    )
    if not ctx.has_model:
        st.warning(ctx.training_error or "No model has been trained.")
        st.info(
            "Evaluation needs historical Pass/Fail outcomes. Upload a dataset with a "
            "`final_result` column, or load the sample dataset from the sidebar."
        )
        return

    trained = ctx.trained
    n_total, n_train, n_test = trained["n_total"], trained["n_train"], trained["n_test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labelled records", f"{n_total:,}")
    c2.metric("Training records", f"{n_train:,}")
    c3.metric("Test records", f"{n_test:,}")
    c4.metric(
        "Fail share", f"{trained['class_counts']['Fail'] / n_total:.0%}",
        help="Share of labelled records with a non-pass outcome (the class the model looks for).",
    )

    st.caption("Models share identical held-out records. Repeated student IDs are kept together in the split and cross-validation. Model selection uses training cross-validation, never held-out F1. Sidebar filters do not change evaluation metrics.")
    st.subheader("Model comparison (test set)")
    summary = ev.metrics_table(trained)
    st.dataframe(
        summary.style.format({c: "{:.3f}" for c in summary.columns if c != "Model"}, na_rep="n/a"),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Precision: of students flagged as Fail, how many really failed. Recall: of students who "
        "really failed, how many were flagged. F1 balances both. Accuracy alone is misleading "
        f"when outcomes are imbalanced: always predicting the most common outcome would score "
        f"{trained['majority_baseline']:.0%}."
    )

    st.subheader("Inspect one model")
    names = list(trained["results"])
    default = names.index(ctx.model_name) if ctx.model_name in names else 0
    chosen = st.selectbox("Model", names, index=default, key="evaluation_model")
    result = trained["results"][chosen]
    metrics = result["metrics"]

    left, right = st.columns(2)
    with left:
        st.pyplot(ev.confusion_matrix_figure(result["confusion_matrix"], chosen), clear_figure=True)
        parts = ev.confusion_breakdown(result["confusion_matrix"])
        st.caption(
            f"Of {n_test} test students: {parts['caught_fail']} non-pass correctly flagged, "
            f"{parts['missed_fail']} non-pass missed, {parts['false_alarm']} flagged but passed, "
            f"{parts['true_pass']} passes correctly left unflagged."
        )
    with right:
        chart(
            ev.importance_figure(result["importance"], chosen),
            "Larger bars mean the feature has more influence on the model's estimate.",
        )
    st.write(
        f"**{chosen}:** accuracy {metrics['accuracy']:.1%}, precision {metrics['precision']:.1%}, "
        f"recall {metrics['recall']:.1%}, F1 {metrics['f1']:.3f}"
        + (f", ROC AUC {metrics['roc_auc']:.3f}." if metrics["roc_auc"] is not None else ".")
    )

    st.subheader("What the risk categories mean on the test set")
    table(
        ev.threshold_table(trained, chosen, ctx.moderate, ctx.high),
        column_config={
            "Actual non-pass rate": st.column_config.NumberColumn(
                "Actual non-pass rate", format="percent",
            ),
        },
    )
    st.caption(
        "If the thresholds are useful, the actual non-pass rate should rise from Low to High. "
        "With few test students per category, these rates are rough."
    )

    st.subheader("Cross-validation (training data only)")
    if trained["cross_validation"]:
        table(pd.DataFrame(trained["cross_validation"]).round(3))
        st.caption("Mean and standard deviation across folds. Large deviations signal unstable results.")
    else:
        st.info(trained["cv_note"])

    st.subheader("How the models were built")
    st.markdown(
        f"- **Target:** `final_result = Fail` is the positive class (1); `Pass` is 0.\n"
        f"- **Features used:** {', '.join(trained['features'])}.\n"
        f"- **Excluded from inputs:** {', '.join(trained['excluded_columns'])}. Identifiers, "
        "names and group labels (department, year, section) are left out to avoid leakage "
        "and proxy bias; the outcome itself is the target.\n"
        f"- **Split:** {1 - trained['test_size']:.0%} training / {trained['test_size']:.0%} test, "
        f"stratified by outcome, random_state={trained['random_state']}.\n"
        "- **Pipeline:** median imputation, standard scaling, then the classifier, all fitted on "
        "training data only.\n"
        "- **Imbalance:** balanced class weights for Logistic Regression, Decision Tree and Random Forest; KNN uses unweighted neighbours."
    )

    st.subheader("Limitations and warnings")
    for message in ev.evaluation_warnings(trained, ctx.using_sample):
        st.warning(message)
