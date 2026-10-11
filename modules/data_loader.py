"""Loading and validating student CSV files.

Concept (Principles of Data Science): *data acquisition and validation*. Data is
checked for structural problems (empty file, missing columns) before any
analysis is attempted, and errors are reported in plain language.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from modules.config import ID_COLUMN, NAME_COLUMN, REQUIRED_COLUMNS

SUGGESTED_COLUMNS = [
    "student_id", "student_name", "department", "year", "section",
    "attendance_percentage", "internal_marks", "assignment_score",
    "previous_semester_marks", "final_result",
]


class DataValidationError(ValueError):
    """Raised when an input CSV cannot support the application's views."""


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, lower-case, snake_case column names."""
    names = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in frame.columns]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise DataValidationError(
            "Column names must be unique (after trimming and lower-casing). "
            "Repeated: " + ", ".join(duplicates)
        )
    standardized = frame.copy()
    standardized.columns = names
    return standardized


def validate_dataset(frame: pd.DataFrame) -> list[str]:
    """Check the structure of a dataset.

    Raises ``DataValidationError`` for problems that make analysis impossible
    and returns a list of non-fatal warnings otherwise.
    """
    if frame is None or len(frame.columns) == 0:
        raise DataValidationError("The CSV has no columns.")
    if frame.empty:
        raise DataValidationError(
            "The CSV contains a header but no student records. Add at least one data row."
        )
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(
            "Required column(s) missing: " + ", ".join(missing)
            + ". Expected columns: " + ", ".join(SUGGESTED_COLUMNS) + "."
        )
    warnings: list[str] = []
    if frame[ID_COLUMN].isna().all():
        raise DataValidationError("student_id contains no values; every record needs an identifier.")
    for column in REQUIRED_COLUMNS[1:]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == 0:
            raise DataValidationError(
                f"Column '{column}' has no numeric values, so it cannot be analysed."
            )
    if frame[ID_COLUMN].isna().any():
        warnings.append(f"{int(frame[ID_COLUMN].isna().sum())} record(s) have no student_id.")
    if len(frame) < 10:
        warnings.append("Fewer than 10 records: charts will work but statistics will be unreliable.")
    return warnings


def read_csv_source(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    """Read a CSV from a path, raw bytes or an uploaded file object.

    Tries UTF-8 (with and without BOM) and falls back to Latin-1 so files saved
    from Excel still load. All values are read as given; type conversion
    happens later in the cleaning module so changes can be reported.
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raw = source.read()
    if not raw or not raw.strip():
        raise DataValidationError("The CSV file is empty.")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except pd.errors.EmptyDataError as exc:
            raise DataValidationError("The CSV file is empty.") from exc
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.ParserError as exc:
            raise DataValidationError(f"The file could not be parsed as CSV: {exc}") from exc
    raise DataValidationError(f"The file encoding is not supported: {last_error}")


def load_csv(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    """Read, standardise and validate a CSV. Raises ``DataValidationError``."""
    frame = standardize_columns(read_csv_source(source))
    validate_dataset(frame)
    return frame


def anonymize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace identifiers with neutral labels and drop student names.

    Labels follow the row order of the prepared dataset, so they stay the same
    when filters are applied. Use this before displaying or exporting data.
    """
    result = frame.copy()
    if ID_COLUMN in result.columns:
        result[ID_COLUMN] = [f"ANON-{i + 1:04d}" for i in range(len(result))]
    if NAME_COLUMN in result.columns:
        result = result.drop(columns=NAME_COLUMN)
    return result
