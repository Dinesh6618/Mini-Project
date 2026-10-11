"""Shared constants for the whole application.

Keeping column names, valid ranges and default thresholds in one place means the
cleaning, analysis and prediction modules always agree with each other.
"""

from __future__ import annotations

import pandas as pd

# --- Column definitions -----------------------------------------------------
ID_COLUMN = "student_id"
NAME_COLUMN = "student_name"
TARGET_COLUMN = "final_result"

REQUIRED_COLUMNS = [
    "student_id",
    "attendance_percentage",
    "internal_marks",
    "assignment_score",
    "previous_semester_marks",
]

GROUP_COLUMNS = ["department", "year", "section"]

# Core numeric columns and the range of values that are considered valid.
CORE_NUMERIC_RANGES = {
    "attendance_percentage": (0, 100),
    "internal_marks": (0, 100),
    "assignment_score": (0, 100),
    "previous_semester_marks": (0, 100),
}
DEFAULT_RANGE = (0, 100)  # used for subject-mark columns such as statistics_marks

CORE_FEATURES = [
    "attendance_percentage",
    "internal_marks",
    "assignment_score",
    "previous_semester_marks",
]
NON_SUBJECT_MARK_COLUMNS = {"internal_marks", "previous_semester_marks"}

# Human readable names used for chart labels and form fields.
FRIENDLY_NAMES = {
    "attendance_percentage": "Attendance (%)",
    "internal_marks": "Internal marks",
    "assignment_score": "Assignment score",
    "previous_semester_marks": "Previous semester marks",
}

# --- Target definition ------------------------------------------------------
# The model predicts "historical non-pass" (positive class = 1).
PASS_LABELS = {"pass", "passed", "p"}
FAIL_LABELS = {"fail", "failed", "not passed", "not_passed", "f"}
PASS_TEXT = "Pass"
FAIL_TEXT = "Fail"

# --- Risk categories --------------------------------------------------------
RISK_LOW = "Low Risk"
RISK_MODERATE = "Moderate Risk"
RISK_HIGH = "High Risk"
RISK_UNSCORED = "Not scored"
RISK_ORDER = [RISK_LOW, RISK_MODERATE, RISK_HIGH, RISK_UNSCORED]

# Models use class_weight="balanced", which pushes scores for the minority
# (non-pass) class upward: a typical student scores well below 0.5 and a student
# the model finds as likely to fail as to pass scores about 0.5. The defaults
# therefore start Moderate at 0.50 rather than at the base failure rate.
DEFAULT_MODERATE_THRESHOLD = 0.50
DEFAULT_HIGH_THRESHOLD = 0.75

# Simple indicator thresholds used for recommendations and for the "needs
# follow-up" count when no trained model is available.
DEFAULT_INDICATORS = {
    "attendance_percentage": 75.0,
    "internal_marks": 40.0,
    "assignment_score": 50.0,
    "decline_points": 10.0,
}

# --- Model training ---------------------------------------------------------
MIN_LABELED_ROWS = 40          # below this the app refuses to train
MIN_PER_CLASS = 8              # minimum examples of each outcome
CV_MIN_TRAIN_ROWS = 80         # cross-validation only above this size
TEST_SIZE = 0.25
RANDOM_STATE = 42


def subject_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric subject-mark columns, e.g. ``statistics_marks``."""
    return [
        column
        for column in frame.columns
        if column.endswith("_marks")
        and column not in NON_SUBJECT_MARK_COLUMNS
        and column != TARGET_COLUMN
    ]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns eligible as model inputs.

    Only academic indicators that exist *before* the final result are used.
    Identifiers, names, group labels (department/year/section) and the target
    itself are deliberately excluded to avoid leakage and proxy bias.
    """
    features = [column for column in CORE_FEATURES if column in frame.columns]
    features.extend(
        column for column in subject_columns(frame)
        if pd.api.types.is_numeric_dtype(frame[column])
    )
    return list(dict.fromkeys(features))


def pretty(column: str) -> str:
    """Readable label for a column name."""
    if column in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[column]
    return column.removesuffix("_marks").replace("_", " ").title() + (
        " marks" if column.endswith("_marks") else ""
    )
