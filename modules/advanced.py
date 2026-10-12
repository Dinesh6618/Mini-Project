"""Data-backed progress, quality, comparisons and held-out group audits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from modules.config import CORE_FEATURES, feature_columns
from modules.prediction import outcome_to_target


def quality_assessment(raw):
    data = raw.replace(r"^\s*$", np.nan, regex=True)
    numeric = feature_columns(data)
    # Subject columns can be strings before cleaning.
    numeric = list(dict.fromkeys(numeric + [c for c in data if c.endswith("_marks")]))
    cells = max(1, data.size)
    missing = int(data.isna().sum().sum())
    invalid = sum(int((data[c].notna() & ~pd.to_numeric(data[c], errors="coerce").between(0, 100)).sum()) for c in numeric)
    duplicates = int(data.duplicated().sum())
    unique = data.drop_duplicates()
    keys = ["student_id"] + (["semester"] if "semester" in data else [])
    inconsistent = int((unique["student_id"].notna() & unique.duplicated(keys, keep=False)).sum())
    if "final_result" in data:
        invalid += int((data.final_result.notna() & outcome_to_target(data.final_result).isna()).sum())
    components = pd.DataFrame([
        ["Missing cells", missing, cells],
        ["Invalid academic values or outcome labels", invalid, cells],
        ["Extra exact duplicate rows", duplicates, max(1, len(data))],
        ["Conflicting student / semester records", inconsistent, max(1, len(unique))],
    ], columns=["Check", "Issues", "Denominator"])
    components["Issue rate"] = components.Issues / components.Denominator
    score = 100 * (1 - components["Issue rate"].mean())
    return float(np.clip(score, 0, 100)), components


def semester_progress(frame):
    if "semester" not in frame:
        raise ValueError("Upload a semester column with positive integer semester numbers, one row per student per semester.")
    data = frame.copy()
    order = pd.to_numeric(data.semester, errors="coerce")
    if order.isna().any() or (order < 1).any() or (order % 1 != 0).any():
        raise ValueError("Semester values must be positive integers (1, 2, 3, ...); ambiguous dates or labels are not ordered automatically.")
    data["semester"] = order.astype(int)
    if data.student_id.isna().any():
        raise ValueError("Progress requires a student_id on every record.")
    if data.duplicated(["student_id", "semester"]).any():
        raise ValueError("Conflicting records exist for the same student and semester. Resolve them before comparing progress.")
    data = data.sort_values(["student_id", "semester"])
    for column in CORE_FEATURES:
        if column in data:
            data[column + "_change"] = data.groupby("student_id")[column].diff()
    return data


def class_comparison(frame, group, subjects):
    if group not in frame or not subjects:
        return pd.DataFrame(), pd.DataFrame()
    means = frame.groupby(group, observed=True)[subjects].mean()
    counts = frame.groupby(group, observed=True)[subjects].count()
    return means, counts


def fairness_audit(frame, trained, model_name, attribute, minimum=10):
    if attribute not in {"department", "year", "section"} or attribute not in frame:
        raise ValueError("Choose an available academic grouping attribute.")
    result = trained["results"][model_name]
    test = frame.loc[trained["test_index"], [attribute, "student_id"]].copy()
    test["actual"] = result["y_true"]
    test["predicted"] = result["y_pred"]
    rows = []
    for group, part in test.dropna(subset=[attribute]).groupby(attribute, observed=True):
        n_students = part.student_id.nunique()
        # Suppress even the group name when the group is too small.
        if n_students < minimum:
            continue
        actual, pred = part.actual, part.predicted
        negatives, positives = int((actual == 0).sum()), int((actual == 1).sum())
        neg_students = part.loc[actual == 0, "student_id"].nunique()
        pos_students = part.loc[actual == 1, "student_id"].nunique()
        rows.append({"Group": str(group), "Students": n_students, "Test records": len(part),
                     "Error rate": float((actual != pred).mean()),
                     "False positive rate": float(((actual == 0) & (pred == 1)).sum() / negatives) if neg_students >= minimum else np.nan,
                     "False negative rate": float(((actual == 1) & (pred == 0)).sum() / positives) if pos_students >= minimum else np.nan})
    return pd.DataFrame(rows)
