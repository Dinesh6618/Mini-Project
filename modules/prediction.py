"""Academic risk prediction with scikit-learn pipelines.

Concepts (Principles of Data Science): *supervised learning*, *train/test
split*, *preprocessing pipelines*, *class imbalance*, *cross-validation* and
*data leakage*.

Target definition
-----------------
``final_result`` is converted to a 0/1 target: ``Fail`` -> 1 (historical
non-pass, the "positive" class the model tries to find) and ``Pass`` -> 0.
The target column, identifiers, names and group labels (department, year,
section) are never used as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from modules.config import (
    CV_MIN_TRAIN_ROWS,
    FAIL_TEXT,
    MIN_LABELED_ROWS,
    MIN_PER_CLASS,
    RANDOM_STATE,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    RISK_UNSCORED,
    TARGET_COLUMN,
    TEST_SIZE,
    feature_columns,
)
from modules.data_cleaning import normalize_outcome


class ModelTrainingError(ValueError):
    """Raised when the available labelled data cannot support a valid model."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def outcome_to_target(series: pd.Series) -> pd.Series:
    """Convert outcome labels to 1 (Fail / non-pass), 0 (Pass) or NaN (unknown)."""
    normalized = series.map(normalize_outcome)
    return normalized.map({FAIL_TEXT: 1.0, "Pass": 0.0})


def risk_category(probability: float, moderate: float, high: float) -> str:
    """Translate a probability into a category using transparent thresholds."""
    if probability is None or pd.isna(probability):
        return RISK_UNSCORED
    if probability >= high:
        return RISK_HIGH
    if probability >= moderate:
        return RISK_MODERATE
    return RISK_LOW


def build_pipeline(estimator, features: list[str]) -> Pipeline:
    """Median imputation -> standard scaling -> classifier.

    Because imputation and scaling live inside the pipeline, they are fitted on
    training data only (no information leaks from the test set).
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer([("numeric", numeric, features)], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def candidate_models() -> dict:
    """The two interpretable baseline models required for the project."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=4, min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    }


def _metrics(y_true, y_pred, y_prob) -> dict:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": None,
    }
    if len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return result


def _importance_table(pipeline: Pipeline, features: list[str], name: str) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        kind = "Coefficient (standardised units; positive raises estimated risk)"
    else:
        values = estimator.feature_importances_
        kind = "Impurity-based importance (0-1; direction not shown)"
    table = pd.DataFrame({"Feature": features, "Value": values, "Kind": kind})
    table["Magnitude"] = table["Value"].abs()
    return table.sort_values("Magnitude", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_models(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train and evaluate Logistic Regression and Decision Tree models.

    Raises ``ModelTrainingError`` with a user-friendly message when the data is
    unsuitable (no labels, too few rows, or too few examples of one outcome).
    """
    if TARGET_COLUMN not in frame.columns:
        raise ModelTrainingError(
            f"The dataset has no '{TARGET_COLUMN}' column, so no historical outcomes are "
            "available. A predictive model cannot be trained honestly without them."
        )
    features = feature_columns(frame)
    if not features:
        raise ModelTrainingError("No numeric academic feature columns were found.")

    target = outcome_to_target(frame[TARGET_COLUMN])
    labelled = target.notna()
    n_unlabelled = int((~labelled).sum())
    if not labelled.any():
        raise ModelTrainingError(
            f"'{TARGET_COLUMN}' has no usable Pass/Fail values. Provide historical outcomes "
            "for past students to train a model."
        )

    X_all = frame.loc[labelled, features].apply(pd.to_numeric, errors="coerce")
    y_all = target.loc[labelled].astype(int)
    usable = X_all.notna().any(axis=1)
    X_all, y_all = X_all.loc[usable], y_all.loc[usable]

    counts = y_all.value_counts().to_dict()
    n_fail, n_pass = counts.get(1, 0), counts.get(0, 0)
    if len(X_all) < MIN_LABELED_ROWS:
        raise ModelTrainingError(
            f"Only {len(X_all)} labelled records are available; at least {MIN_LABELED_ROWS} "
            "are needed for a train/test split that gives meaningful results."
        )
    if min(n_fail, n_pass) < MIN_PER_CLASS:
        raise ModelTrainingError(
            f"The outcomes are too one-sided (Pass={n_pass}, Fail={n_fail}). At least "
            f"{MIN_PER_CLASS} examples of each outcome are needed."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )

    models, results = {}, {}
    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator, features)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
        models[name] = pipeline
        results[name] = {
            "metrics": _metrics(y_test, predicted, probability),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]),
            "importance": _importance_table(pipeline, features, name),
            "y_true": y_test.to_numpy(),
            "y_pred": predicted,
            "y_prob": probability,
        }

    # Cross-validation on the training portion only -------------------------
    cross_validation: list[dict] = []
    min_train_class = int(y_train.value_counts().min())
    cv_note = ""
    if len(X_train) >= CV_MIN_TRAIN_ROWS and min_train_class >= 5:
        folds = min(5, min_train_class)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        for name, estimator in candidate_models().items():
            scores = cross_validate(
                build_pipeline(estimator, features), X_train, y_train, cv=splitter,
                scoring={"precision": "precision", "recall": "recall", "f1": "f1"},
                error_score="raise",
            )
            row = {"Model": name, "Folds": folds}
            for key in ("precision", "recall", "f1"):
                row[f"{key.title()} (mean)"] = float(np.mean(scores[f"test_{key}"]))
                row[f"{key.title()} (std)"] = float(np.std(scores[f"test_{key}"]))
            cross_validation.append(row)
    else:
        cv_note = (
            f"Cross-validation was skipped: it needs at least {CV_MIN_TRAIN_ROWS} training "
            "records and 5 examples of each outcome."
        )

    # Warnings ----------------------------------------------------------------
    notes: list[str] = []
    minority_test = int(min((y_test == 1).sum(), (y_test == 0).sum()))
    if len(X_test) < 50 or minority_test < 10:
        notes.append(
            f"The test set is small ({len(X_test)} records, {minority_test} in the smaller "
            "outcome group). Metrics can change a lot with a different split."
        )
    if min(n_fail, n_pass) / len(y_all) < 0.3:
        notes.append(
            f"Outcomes are imbalanced (Fail = {n_fail / len(y_all):.0%}). Class weights were "
            "used so the minority class is not ignored, and accuracy alone is misleading."
        )
    for name, result in results.items():
        if result["metrics"]["f1"] >= 0.98 and result["metrics"]["accuracy"] >= 0.98:
            notes.append(
                f"{name} scored almost perfectly. On real data this often indicates leakage "
                "or labels generated directly from the inputs; check before trusting it."
            )
    if n_unlabelled:
        notes.append(
            f"{n_unlabelled} record(s) without a Pass/Fail outcome were not used for "
            "training or evaluation."
        )

    return {
        "features": features,
        "models": models,
        "results": results,
        "cross_validation": cross_validation,
        "cv_note": cv_note,
        "notes": notes,
        "n_total": len(X_all),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_unlabelled": n_unlabelled,
        "class_counts": {"Pass": n_pass, "Fail": n_fail},
        "test_size": test_size,
        "random_state": random_state,
        "majority_baseline": float(max(n_fail, n_pass) / len(y_all)),
        "excluded_columns": [
            c for c in frame.columns
            if c not in features and c not in {"source_row", "risk_probability", "risk_category"}
        ],
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def fail_probability(pipeline: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """Probability of the non-pass class for each row of ``data``."""
    column = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(data)[:, column]


def score_students(frame: pd.DataFrame, trained: dict, model_name: str) -> pd.Series:
    """Estimated non-pass probability for every student.

    Students with a known outcome receive an *out-of-fold* probability (from a
    model that never saw that student), so scores are not inflated by
    memorisation. Students without an outcome are scored by the trained model.
    Students with no usable indicators are NaN.
    """
    features = trained["features"]
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    scorable = X.notna().any(axis=1)
    probability = pd.Series(np.nan, index=frame.index, dtype=float)

    target = outcome_to_target(frame[TARGET_COLUMN]) if TARGET_COLUMN in frame else pd.Series(
        np.nan, index=frame.index
    )
    known = scorable & target.notna()
    unknown = scorable & target.isna()

    estimator = candidate_models()[model_name]
    if known.sum() >= MIN_LABELED_ROWS:
        y = target.loc[known].astype(int)
        folds = int(max(2, min(5, y.value_counts().min())))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        pipeline = build_pipeline(clone(estimator), features)
        oof = cross_val_predict(pipeline, X.loc[known], y, cv=splitter, method="predict_proba")
        probability.loc[known] = oof[:, 1]
    if unknown.any():
        probability.loc[unknown] = fail_probability(trained["models"][model_name], X.loc[unknown])
    return probability


def apply_risk_categories(probability: pd.Series, moderate: float, high: float) -> pd.Series:
    """Vectorised version of ``risk_category``."""
    return probability.map(lambda p: risk_category(p, moderate, high))


def predict_single(trained: dict, model_name: str, values: dict[str, float]) -> float:
    """Estimated non-pass probability for one student's entered values."""
    row = pd.DataFrame([values], columns=trained["features"]).apply(pd.to_numeric, errors="coerce")
    return float(fail_probability(trained["models"][model_name], row)[0])


def explain_single(trained: dict, model_name: str, values: dict[str, float]) -> pd.DataFrame | None:
    """Per-feature contribution to the estimate (Logistic Regression only).

    Each contribution is ``coefficient x standardised value`` and is measured in
    log-odds: positive numbers push the estimate towards non-pass. Returns
    ``None`` for models without linear coefficients.
    """
    pipeline = trained["models"][model_name]
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return None
    features = trained["features"]
    row = pd.DataFrame([values], columns=features).apply(pd.to_numeric, errors="coerce")
    standardised = pipeline.named_steps["preprocess"].transform(row)[0]
    contribution = estimator.coef_[0] * standardised
    table = pd.DataFrame({
        "Feature": features,
        "Entered value": [row.iloc[0][f] for f in features],
        "Contribution": contribution,
    })
    return table.reindex(table["Contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
