"""Supportive, explainable academic-support suggestions.

Concept (Principles of Data Science): *interpretability*. Every suggestion comes
from a visible, rule-based threshold on an academic indicator, so it can be
explained to a student or mentor. Suggestions are phrased as offers of help,
and no personal circumstances are inferred from marks or attendance.
"""

from __future__ import annotations

import pandas as pd

from modules.config import DEFAULT_INDICATORS


def academic_recommendations(
    values: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Return supportive suggestions for one student's indicator values.

    ``values`` maps column names to numbers (missing/NaN values are skipped).
    ``thresholds`` can override ``DEFAULT_INDICATORS``.
    """
    limits = {**DEFAULT_INDICATORS, **(thresholds or {})}

    def get(name: str):
        value = values.get(name)
        return None if value is None or pd.isna(value) else float(value)

    suggestions: list[str] = []
    attendance = get("attendance_percentage")
    if attendance is not None and attendance < limits["attendance_percentage"]:
        suggestions.append(
            f"Attendance is below {limits['attendance_percentage']:.0f}%. Review the attendance "
            "record together and ask whether there are any barriers to participation "
            "that the department could help with."
        )
    internal = get("internal_marks")
    if internal is not None and internal < limits["internal_marks"]:
        suggestions.append(
            f"Internal marks are below {limits['internal_marks']:.0f}. Offer revision sessions, "
            "extra practice questions and a chance to talk through difficult topics."
        )
    assignment = get("assignment_score")
    if assignment is not None and assignment < limits["assignment_score"]:
        suggestions.append(
            f"Assignment scores are below {limits['assignment_score']:.0f}. Go through the "
            "assignment requirements together and share a worked example or feedback."
        )
    previous = get("previous_semester_marks")
    if (
        internal is not None and previous is not None
        and internal + limits["decline_points"] < previous
    ):
        suggestions.append(
            f"Marks have dropped by more than {limits['decline_points']:.0f} points since the "
            "previous semester. Consider a supportive conversation with the student and "
            "an academic mentor to understand what would help."
        )
    for column, value in values.items():
        if column.endswith("_marks") and column not in {"internal_marks", "previous_semester_marks"}:
            number = get(column)
            if number is not None and number < limits["internal_marks"]:
                subject = column.removesuffix("_marks").replace("_", " ")
                suggestions.append(
                    f"Marks in {subject} are below {limits['internal_marks']:.0f}. "
                    f"Subject-specific tutoring or office hours for {subject} may help."
                )
    return suggestions


def indicator_flags(frame: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Boolean table of which indicator thresholds each student falls below."""
    limits = {**DEFAULT_INDICATORS, **(thresholds or {})}
    flags = pd.DataFrame(index=frame.index)
    if "attendance_percentage" in frame:
        flags["Low attendance"] = frame["attendance_percentage"] < limits["attendance_percentage"]
    if "internal_marks" in frame:
        flags["Low internal marks"] = frame["internal_marks"] < limits["internal_marks"]
    if "assignment_score" in frame:
        flags["Low assignment score"] = frame["assignment_score"] < limits["assignment_score"]
    if {"internal_marks", "previous_semester_marks"}.issubset(frame.columns):
        flags["Declining marks"] = (
            frame["internal_marks"] + limits["decline_points"] < frame["previous_semester_marks"]
        )
    return flags.fillna(False).astype(bool)


def needs_follow_up(frame: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.Series:
    """True for students with at least one indicator below its threshold."""
    flags = indicator_flags(frame, thresholds)
    if flags.empty:
        return pd.Series(False, index=frame.index)
    return flags.any(axis=1)


def recommendation_summary(frame: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Number of students per indicator, for the dashboard."""
    flags = indicator_flags(frame, thresholds)
    return pd.DataFrame({
        "Indicator": flags.columns,
        "Students": [int(flags[c].sum()) for c in flags.columns],
        "Share of students": [float(flags[c].mean()) if len(flags) else 0.0 for c in flags.columns],
    })
