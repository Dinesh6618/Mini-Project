"""About Project page: summary, concept map, data card and module guide."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from views.common import page_header, table

CONCEPTS = pd.DataFrame([
    ("Data acquisition and validation", "Data Loader, Student Records", "modules/data_loader.py"),
    ("Data cleaning and preprocessing", "Data Cleaning", "modules/data_cleaning.py"),
    ("Descriptive statistics and distributions", "Dashboard, Exploratory Analysis", "modules/exploratory_analysis.py"),
    ("Correlation and association (not causation)", "Exploratory Analysis", "modules/exploratory_analysis.py"),
    ("Outlier detection (IQR rule)", "Exploratory Analysis", "modules/exploratory_analysis.py"),
    ("Supervised classification", "Risk Prediction", "modules/prediction.py"),
    ("Train/test split, pipelines, class imbalance", "Risk Prediction", "modules/prediction.py"),
    ("Cross-validation and data leakage", "Model Evaluation", "modules/prediction.py"),
    ("Precision, recall, F1, confusion matrix", "Model Evaluation", "modules/model_evaluation.py"),
    ("Interpretability and explainable rules", "Risk Prediction, Recommendations", "modules/recommendations.py"),
    ("Data storage (SQLite)", "Student Records", "modules/database.py"),
    ("Ethics, privacy, bias and human oversight", "Data Ethics", "views/ethics.py"),
], columns=["Data Science concept", "Where it appears", "Source file"])

MODULES = pd.DataFrame([
    ("app.py", "Page setup, sidebar, data selection, filters, caching and navigation."),
    ("modules/config.py", "Column names, valid ranges, default thresholds and the target definition."),
    ("modules/data_loader.py", "Reads CSV files, standardises column names, validates structure, anonymises IDs."),
    ("modules/data_cleaning.py", "Type conversion, range checks, duplicates, label normalisation and the change log."),
    ("modules/exploratory_analysis.py", "Statistics, correlations, outliers and the Plotly charts."),
    ("modules/prediction.py", "Pipelines, Logistic Regression and Decision Tree, evaluation, risk scoring."),
    ("modules/model_evaluation.py", "Metric tables, confusion matrix, importance charts, warnings."),
    ("modules/recommendations.py", "Rule-based, supportive suggestions."),
    ("modules/database.py", "Optional local SQLite storage."),
    ("modules/theme.py", "Colours, CSS and shared chart styling."),
    ("views/*.py", "One file per page."),
    ("data/generate_sample_data.py", "Script that produced the synthetic sample dataset."),
], columns=["File", "Responsibility"])


def render(ctx) -> None:
    page_header(
        "About This Project",
        "Student Academic Performance Analysis and Early Risk Prediction Using Data Science "
        "&middot; Principles of Data Science mini project",
    )

    st.subheader("Problem")
    st.write(
        "Colleges hold attendance, internal marks, assignment scores and previous results, but "
        "reviewing them by hand makes it hard to notice students who may be struggling early. "
        "This application analyses such records, shows patterns, and uses machine learning to "
        "suggest which students might benefit from extra academic support."
    )
    st.info(
        "It is a decision-support tool. It does not label, rank or penalise students, and a "
        "person must review every estimate."
    )

    st.subheader("Demonstration summary")
    st.markdown(
        "1. **Load and validate** a CSV (or the built-in synthetic sample).\n"
        "2. **Clean** it with every change recorded in a change log.\n"
        "3. **Explore** distributions, correlations, trends and outliers.\n"
        "4. **Train** Logistic Regression and a Decision Tree on historical outcomes, with a "
        "held-out test set, class weights and cross-validation.\n"
        "5. **Evaluate** honestly with precision, recall, F1 and a confusion matrix.\n"
        "6. **Estimate risk** for individual students, with adjustable thresholds and "
        "supportive, rule-based suggestions.\n"
        "7. **Reflect** on privacy, bias and human oversight."
    )

    st.subheader("Syllabus concepts demonstrated")
    table(CONCEPTS)

    st.subheader("Sample dataset: what it is")
    st.markdown(
        "- `data/sample_students.csv` is **fictional and synthetic**. Names are random "
        "combinations; no real students or college records are included.\n"
        "- Each student's indicators were generated from hidden ability and engagement values "
        "plus random noise.\n"
        "- **Target:** `final_result = Fail` means a historical non-pass (class 1); `Pass` is 0. "
        "The outcome was generated from the indicators with an artificial rule plus random "
        "noise, so model scores on this file are **not** evidence of real-world accuracy.\n"
        "- Quality problems were inserted on purpose: missing values, out-of-range and "
        "non-numeric entries, exact duplicates, a repeated ID, inconsistent labels and 12 "
        "students with no outcome yet.\n"
        "- Regenerate it with `python data/generate_sample_data.py` (fixed random seed)."
    )

    st.subheader("Project files")
    table(MODULES)

    st.subheader("Known limitations")
    st.markdown(
        "- Only two baseline models are compared, with default-style settings.\n"
        "- Results come from one train/test split; they will vary with different data.\n"
        "- Subject columns must end in `_marks` and use a 0-100 scale.\n"
        "- No fairness audit across groups is performed.\n"
        "- Not intended for use on real students without institutional approval and review."
    )
