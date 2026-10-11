"""Data cleaning and preparation with a full audit trail.

Concept (Principles of Data Science): *data preprocessing / data quality*.
Every change made to the data is counted and listed in a change log, so
nothing is modified silently. Missing values are **not** imputed here; they stay
visible, and the model pipeline imputes them later using training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.config import (
    CORE_NUMERIC_RANGES,
    DEFAULT_RANGE,
    FAIL_LABELS,
    FAIL_TEXT,
    ID_COLUMN,
    PASS_LABELS,
    PASS_TEXT,
    TARGET_COLUMN,
    subject_columns,
)

ISSUE_COLUMNS = [
    "Column", "Allowed range", "Missing in input", "Not numeric",
    "Outside range", "Missing after cleaning",
]
LOG_COLUMNS = ["Row", "Student ID", "Column", "Original value", "Action"]


def _csv_row(index) -> int:
    """CSV line number of a record (header is line 1, first record line 2)."""
    return int(index) + 2


def normalize_outcome(value) -> str | None:
    """Map an outcome label to 'Pass' / 'Fail', or ``None`` if blank/unknown."""
    if pd.isna(value):
        return None
    text = str(value).strip().casefold()
    if text in PASS_LABELS:
        return PASS_TEXT
    if text in FAIL_LABELS:
        return FAIL_TEXT
    return None


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a validated student dataset.

    Returns ``(cleaned_frame, report)``. The report contains:

    - ``rows_input`` / ``rows_output``: record counts before and after.
    - ``duplicates_removed``: exact duplicate rows dropped (first kept).
    - ``duplicate_ids``: rows sharing a student_id (kept, flagged for review).
    - ``issues``: per-column table of missing / invalid values.
    - ``operations``: plain-language list of what was done.
    - ``warnings``: things a human should look at.
    - ``change_log``: one row per changed value or removed record.
    - ``dtypes``: data type of each column after cleaning.
    """
    cleaned = frame.drop(columns=["source_row"], errors="ignore").copy()
    operations: list[str] = []
    warnings: list[str] = []
    log: list[dict] = []
    rows_input = len(cleaned)

    def student_id(index) -> object:
        if ID_COLUMN in cleaned.columns and index in cleaned.index:
            return cleaned.at[index, ID_COLUMN]
        return None

    # 1. Trim whitespace in text columns; treat empty strings as missing --------
    text_columns = [
        c for c in cleaned.columns
        if cleaned[c].dtype == object or pd.api.types.is_string_dtype(cleaned[c])
    ]
    trimmed_cells = 0
    blank_cells = 0
    for column in text_columns:
        original = cleaned[column]
        as_text = original.astype("string")
        stripped = as_text.str.strip()
        changed = (original.notna() & (as_text != stripped)).fillna(False).astype(bool)
        trimmed_cells += int(changed.sum())
        for index in cleaned.index[changed]:
            log.append({
                "Row": _csv_row(index),
                "Student ID": cleaned.at[index, ID_COLUMN] if ID_COLUMN in cleaned.columns else None,
                "Column": column, "Original value": repr(str(original.at[index])),
                "Action": f"Spaces trimmed to {str(stripped.at[index])!r}",
            })
        blank = stripped.eq("").fillna(False).astype(bool)
        blank_cells += int(blank.sum())
        cleaned[column] = stripped.astype(object).where(stripped.notna() & ~blank, np.nan)
    if trimmed_cells:
        operations.append(f"Trimmed leading/trailing spaces in {trimmed_cells} text value(s).")
    if blank_cells:
        operations.append(f"Treated {blank_cells} empty text value(s) as missing.")

    # 2. Exact duplicate rows ----------------------------------------------------
    duplicate_mask = cleaned.duplicated(keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    if duplicates_removed:
        for index in cleaned.index[duplicate_mask]:
            log.append({
                "Row": _csv_row(index), "Student ID": student_id(index),
                "Column": "(entire row)", "Original value": "identical to an earlier row",
                "Action": "Row removed (exact duplicate); first occurrence kept",
            })
        cleaned = cleaned.loc[~duplicate_mask].copy()
        operations.append(
            f"Removed {duplicates_removed} exact duplicate row(s); the first occurrence was kept."
        )
        warnings.append(
            f"{duplicates_removed} duplicate row(s) were removed. Confirm the source file "
            "did not repeat records for a legitimate reason."
        )
    else:
        operations.append("No exact duplicate rows were found.")

    # 3. Repeated student IDs (kept, flagged) ------------------------------------
    duplicate_ids = 0
    if ID_COLUMN in cleaned.columns:
        ids = cleaned[ID_COLUMN]
        duplicate_ids = int(ids.notna().mul(ids.duplicated(keep=False)).sum())
    if duplicate_ids:
        operations.append(
            f"Found {duplicate_ids} row(s) sharing a student_id with different values; "
            "they were kept and need manual review."
        )
        warnings.append(
            f"{duplicate_ids} row(s) share a student_id but differ in other columns. "
            "They were NOT merged; check which record is correct."
        )

    # 4. Numeric columns: type conversion and range validation -------------------
    numeric_columns = [c for c in CORE_NUMERIC_RANGES if c in cleaned.columns]
    numeric_columns += [c for c in subject_columns(cleaned) if c not in numeric_columns]
    issue_rows: list[dict] = []
    for column in numeric_columns:
        original = cleaned[column]
        lower, upper = CORE_NUMERIC_RANGES.get(column, DEFAULT_RANGE)
        numeric = pd.to_numeric(original, errors="coerce")
        missing_before = int(original.isna().sum())
        not_numeric = original.notna() & numeric.isna()
        out_of_range = numeric.notna() & ~numeric.between(lower, upper)

        for index in cleaned.index[not_numeric]:
            log.append({
                "Row": _csv_row(index), "Student ID": student_id(index), "Column": column,
                "Original value": original.at[index],
                "Action": "Not a number; set to missing",
            })
        for index in cleaned.index[out_of_range]:
            log.append({
                "Row": _csv_row(index), "Student ID": student_id(index), "Column": column,
                "Original value": original.at[index],
                "Action": f"Outside {lower}-{upper}; set to missing",
            })
        numeric = numeric.mask(out_of_range)
        cleaned[column] = numeric.astype(float)
        missing_after = int(cleaned[column].isna().sum())

        if missing_before or not_numeric.any() or out_of_range.any():
            issue_rows.append({
                "Column": column,
                "Allowed range": f"{lower}-{upper}",
                "Missing in input": missing_before,
                "Not numeric": int(not_numeric.sum()),
                "Outside range": int(out_of_range.sum()),
                "Missing after cleaning": missing_after,
            })
        if not_numeric.any() or out_of_range.any():
            operations.append(
                f"{column}: set {int(not_numeric.sum())} non-numeric and "
                f"{int(out_of_range.sum())} out-of-range value(s) to missing "
                f"(allowed {lower}-{upper})."
            )
        share = missing_after / max(len(cleaned), 1)
        if share > 0.2:
            warnings.append(
                f"{column} is missing for {share:.0%} of records; results using it may be unreliable."
            )
    if numeric_columns:
        operations.append(
            "Converted to numeric: " + ", ".join(numeric_columns) + "."
        )

    # 5. Group columns (year) -----------------------------------------------------
    if "year" in cleaned.columns:
        converted = pd.to_numeric(cleaned["year"], errors="coerce")
        if converted.notna().sum() == cleaned["year"].notna().sum():
            cleaned["year"] = converted.astype("Int64")
            operations.append("Converted year to whole numbers.")
        else:
            warnings.append("Some 'year' values are not numbers; the column was kept as text.")

    # 6. Outcome labels -----------------------------------------------------------
    if TARGET_COLUMN in cleaned.columns:
        original = cleaned[TARGET_COLUMN]
        normalized = original.map(normalize_outcome)
        unrecognized = original.notna() & normalized.isna()
        changed = original.notna() & normalized.notna() & (original.astype(str) != normalized)
        for index in cleaned.index[unrecognized]:
            log.append({
                "Row": _csv_row(index), "Student ID": student_id(index), "Column": TARGET_COLUMN,
                "Original value": original.at[index],
                "Action": "Unrecognised outcome label; set to missing",
            })
        if changed.any():
            for index in cleaned.index[changed]:
                log.append({
                    "Row": _csv_row(index), "Student ID": student_id(index),
                    "Column": TARGET_COLUMN, "Original value": original.at[index],
                    "Action": f"Label standardised to '{normalized.at[index]}'",
                })
            operations.append(
                f"Standardised {int(changed.sum())} {TARGET_COLUMN} label(s) to Pass/Fail "
                "(for example 'pass', 'FAIL', 'Passed')."
            )
        if unrecognized.any():
            warnings.append(
                f"{int(unrecognized.sum())} {TARGET_COLUMN} value(s) were not recognised as "
                "Pass/Fail and were set to missing."
            )
        cleaned[TARGET_COLUMN] = normalized
        n_missing = int(cleaned[TARGET_COLUMN].isna().sum())
        if n_missing:
            operations.append(
                f"{n_missing} record(s) have no {TARGET_COLUMN}; they are kept, but cannot be "
                "used as training examples."
            )
            issue_rows.append({
                "Column": TARGET_COLUMN, "Allowed range": "Pass / Fail",
                "Missing in input": int(original.isna().sum()),
                "Not numeric": 0, "Outside range": int(unrecognized.sum()),
                "Missing after cleaning": n_missing,
            })

    # 7. Remaining missing values in other columns --------------------------------
    handled = {row["Column"] for row in issue_rows}
    for column in cleaned.columns:
        if column in handled:
            continue
        missing = int(cleaned[column].isna().sum())
        if missing:
            issue_rows.append({
                "Column": column, "Allowed range": "-", "Missing in input": missing,
                "Not numeric": 0, "Outside range": 0, "Missing after cleaning": missing,
            })

    operations.append(
        "Missing values were left as missing (no imputation). The prediction pipeline "
        "fills them with medians learned from the training split only."
    )

    cleaned = cleaned.reset_index(drop=False).rename(columns={"index": "source_row"})
    cleaned["source_row"] = cleaned["source_row"].map(_csv_row)

    issues = pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)
    change_log = pd.DataFrame(log, columns=LOG_COLUMNS)
    # Mixed numbers and text in one column cannot be displayed reliably; show as text.
    change_log["Original value"] = change_log["Original value"].map(
        lambda v: "" if pd.isna(v) else str(v)
    )
    change_log["Student ID"] = change_log["Student ID"].astype("string")
    dtypes = pd.DataFrame({
        "Column": cleaned.columns,
        "Type after cleaning": [str(t) for t in cleaned.dtypes],
        "Missing values": [int(cleaned[c].isna().sum()) for c in cleaned.columns],
    })
    return cleaned, {
        "rows_input": rows_input,
        "rows_output": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "duplicate_ids": duplicate_ids,
        "issues": issues,
        "operations": operations,
        "warnings": warnings,
        "change_log": change_log,
        "dtypes": dtypes,
    }
