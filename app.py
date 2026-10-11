"""Student Academic Performance Analysis and Early Risk Prediction.

Run with:  streamlit run app.py

This file only wires things together: it loads data, applies cleaning, trains
the models (cached), builds the sidebar filters and routes to one view per page.
The analysis logic lives in ``modules/`` and the page layouts in ``views/``.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from modules import database
from modules.config import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_INDICATORS,
    DEFAULT_MODERATE_THRESHOLD,
    GROUP_COLUMNS,
    RISK_ORDER,
)
from modules.data_cleaning import clean_dataset
from modules.data_loader import (
    DataValidationError,
    anonymize_frame,
    load_csv,
    standardize_columns,
    validate_dataset,
)
from modules.prediction import (
    ModelTrainingError,
    apply_risk_categories,
    score_students,
    train_models,
)
from modules.theme import CSS
from views import about, cleaning, dashboard, ethics, evaluation, exploration, records, risk_prediction
from views.context import AppContext

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_students.csv"

PAGES = {
    "Dashboard": dashboard.render,
    "Student Records": records.render,
    "Data Cleaning": cleaning.render,
    "Exploratory Analysis": exploration.render,
    "Risk Prediction": risk_prediction.render,
    "Model Evaluation": evaluation.render,
    "Data Ethics": ethics.render,
    "About Project": about.render,
}

SOURCE_SAMPLE = "Sample dataset (synthetic)"
SOURCE_UPLOAD = "Upload a CSV file"
SOURCE_DATABASE = "Saved local database"

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading data...")
def read_upload(content: bytes) -> pd.DataFrame:
    return load_csv(content)


@st.cache_data(show_spinner="Cleaning data...")
def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return clean_dataset(raw)


@st.cache_resource(show_spinner="Training models...")
def train_cached(cleaned: pd.DataFrame) -> tuple[dict | None, str | None]:
    """Train once per dataset. Failures are returned (and cached), not raised."""
    try:
        return train_models(cleaned), None
    except ModelTrainingError as exc:
        return None, str(exc)


@st.cache_data(show_spinner="Estimating risk...")
def score_cached(cleaned: pd.DataFrame, _trained: dict, model_name: str) -> pd.Series:
    return score_students(cleaned, _trained, model_name)


# ---------------------------------------------------------------------------
# Sidebar sections
# ---------------------------------------------------------------------------
def choose_data() -> tuple[pd.DataFrame, str, bool]:
    """Sidebar data-source selection. Returns (raw frame, label, is_sample)."""
    st.sidebar.markdown("**Data**")
    options = [SOURCE_SAMPLE, SOURCE_UPLOAD]
    if database.database_exists():
        options.append(SOURCE_DATABASE)
    source = st.sidebar.radio("Dataset", options, key="source", label_visibility="collapsed")

    try:
        if source == SOURCE_SAMPLE:
            st.sidebar.caption(
                "Fictional records generated for demonstration. Not real student data."
            )
            return load_csv(SAMPLE_PATH), "Sample dataset (fictional, synthetic)", True
        if source == SOURCE_DATABASE:
            frame = standardize_columns(database.load_dataset())
            validate_dataset(frame)
            return frame, "Saved local database", False
        uploaded = st.sidebar.file_uploader("Student CSV", type=["csv"], key="upload")
        with st.sidebar.expander("Expected CSV columns"):
            st.markdown(
                "**Required:** `student_id`, `attendance_percentage`, `internal_marks`, "
                "`assignment_score`, `previous_semester_marks`\n\n"
                "**Optional:** `student_name`, `department`, `year`, `section`, "
                "`final_result` (Pass/Fail) and subject columns ending in `_marks`.\n\n"
                "Scores and attendance use a 0-100 scale."
            )
        if uploaded is None:
            st.sidebar.info("No file uploaded yet, so the sample dataset is shown.")
            return (
                load_csv(SAMPLE_PATH),
                "Sample dataset (fictional, synthetic) - no file uploaded yet",
                True,
            )
        # The file name is placed in HTML captions, so escape it.
        return read_upload(uploaded.getvalue()), f"Uploaded file: {html.escape(uploaded.name)}", False
    except DataValidationError as exc:
        st.title("This file could not be used")
        st.error(str(exc))
        st.info("Fix the file and upload it again, or switch to the sample dataset in the sidebar.")
        st.stop()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.title("This file could not be read")
        st.error(f"Reading the data failed: {exc}")
        st.stop()
    raise RuntimeError("unreachable")  # pragma: no cover


def risk_settings(trained: dict | None) -> tuple[str | None, float, float, dict]:
    """Sidebar controls for thresholds and the model used for estimates."""
    with st.sidebar.expander("Risk settings", expanded=False):
        model_name = None
        if trained is not None:
            names = list(trained["results"])
            best = max(names, key=lambda n: trained["results"][n]["metrics"]["f1"])
            model_name = st.selectbox(
                "Model for risk estimates", names, index=names.index(best), key="risk_model",
                help="Defaults to the model with the higher F1 on the held-out test set.",
            )
        moderate = st.slider(
            "Moderate risk from probability", 0.10, 0.80, DEFAULT_MODERATE_THRESHOLD, 0.05,
            key="moderate",
        )
        high = st.slider(
            "High risk from probability", 0.20, 0.95, DEFAULT_HIGH_THRESHOLD, 0.05, key="high",
        )
        if high <= moderate:
            high = min(0.95, moderate + 0.05)
            st.caption(f"High threshold raised to {high:.0%} so it stays above Moderate.")
        st.markdown("**Indicator thresholds**")
        indicators = {
            "attendance_percentage": float(st.number_input(
                "Low attendance below (%)", 0.0, 100.0, DEFAULT_INDICATORS["attendance_percentage"], 1.0)),
            "internal_marks": float(st.number_input(
                "Low internal marks below", 0.0, 100.0, DEFAULT_INDICATORS["internal_marks"], 1.0)),
            "assignment_score": float(st.number_input(
                "Low assignment score below", 0.0, 100.0, DEFAULT_INDICATORS["assignment_score"], 1.0)),
            "decline_points": float(st.number_input(
                "Marks drop of more than", 0.0, 50.0, DEFAULT_INDICATORS["decline_points"], 1.0)),
        }
    return model_name, float(moderate), float(high), indicators


def build_filters(scored: pd.DataFrame) -> pd.Series:
    """Sidebar multiselect filters; returns a boolean mask over ``scored``."""
    mask = pd.Series(True, index=scored.index)
    filter_columns = [c for c in GROUP_COLUMNS if c in scored.columns]
    if "risk_category" in scored.columns:
        filter_columns.append("risk_category")
    if not filter_columns:
        return mask
    st.sidebar.markdown("**Filters**")
    for column in filter_columns:
        values = scored[column].dropna()
        if column == "risk_category":
            options = [c for c in RISK_ORDER if c in set(values)]
        else:
            options = sorted(values.astype(str).unique().tolist())
        if not options:
            continue
        signature = abs(hash(tuple(options)))
        chosen = st.sidebar.multiselect(
            column.replace("_", " ").title(), options, default=options,
            key=f"filter_{column}_{signature}",
        )
        if len(chosen) < len(options):
            mask &= scored[column].astype(str).isin(chosen)
    return mask


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
st.sidebar.title("Student Performance")
st.sidebar.caption("Academic analysis and early-support estimates")
# Optional deep link, e.g. http://localhost:8501/?page=Risk%20Prediction
if "page" not in st.session_state and st.query_params.get("page") in PAGES:
    st.session_state["page"] = st.query_params["page"]
page = st.sidebar.radio("Navigation", list(PAGES), key="page")
st.sidebar.divider()

raw, source_label, using_sample = choose_data()
try:
    cleaned, report = prepare(raw)
except Exception as exc:  # unexpected data problem: report, don't crash
    st.title("Data preparation failed")
    st.error(f"The data could not be prepared: {exc}")
    st.stop()

anonymize = st.sidebar.checkbox(
    "Use anonymous student IDs", value=True, key="anonymize",
    help="Hides names and replaces IDs with ANON-#### labels in tables and downloads.",
)
if report["warnings"]:
    st.sidebar.warning("Data-quality warnings are listed on the Data Cleaning page.")

trained, training_error = train_cached(cleaned)
model_name, moderate, high, indicators = risk_settings(trained)

scored = cleaned.copy()
if trained is not None and model_name is not None:
    probability = score_cached(cleaned, trained, model_name)
    scored["risk_probability"] = probability
    scored["risk_category"] = apply_risk_categories(probability, moderate, high)

mask = build_filters(scored)
display = anonymize_frame(scored) if anonymize else scored
view = display[mask]

context = AppContext(
    raw=raw,
    cleaned=cleaned,
    report=report,
    source_label=source_label,
    using_sample=using_sample,
    anonymize=anonymize,
    scored=scored,
    view=view,
    trained=trained,
    training_error=training_error,
    model_name=model_name,
    moderate=moderate,
    high=high,
    indicators=indicators,
    filters_active=bool((~mask).any()),
)

try:
    PAGES[page](context)
except Exception as exc:  # keep the sidebar usable if one page fails
    st.error(f"This page could not be displayed: {exc}")
    st.exception(exc)
