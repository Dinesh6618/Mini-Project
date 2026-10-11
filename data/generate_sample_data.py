"""Generate the FICTIONAL sample dataset used by the application.

Run from the project folder:

    python data/generate_sample_data.py

How the data is made
--------------------
* 300 imaginary students; names are random combinations of common first names
  and surnames and do not refer to real people.
* Each student gets a hidden "ability" and "engagement" value. Attendance,
  internal marks, assignment score, previous-semester marks and three subject
  marks are produced from those values plus random noise.
* ``final_result`` is an ARTIFICIAL outcome: the chance of "Fail" rises when
  attendance, internal marks, assignment score and previous marks are low, and
  a random term is added so the outcome is not a perfect rule. Because the
  outcome was generated from the input columns, any model trained on this file
  will look better than it would on real student records.
* A few data-quality problems are inserted on purpose (missing values, an
  out-of-range mark, a non-numeric entry, exact duplicate rows, inconsistent
  label spelling, extra spaces, a repeated ID, and 12 students with no
  outcome yet) so the cleaning page has something to demonstrate.

Target definition used by the app: final_result == "Fail" is a historical
non-pass (class 1); "Pass" is class 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2024
N_STUDENTS = 300
OUTPUT = Path(__file__).resolve().parent / "sample_students.csv"

FIRST_NAMES = [
    "Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Bhavna", "Deepak", "Divya", "Farhan",
    "Gayathri", "Harini", "Ishaan", "Jaya", "Karthik", "Lakshmi", "Manoj", "Meera", "Naveen",
    "Nisha", "Pooja", "Pranav", "Rahul", "Riya", "Sanjay", "Shreya", "Sneha", "Suresh",
    "Tanvi", "Varun", "Vidya", "Yash", "Zoya", "Kavya", "Rohan", "Sahana", "Vikram",
]
LAST_NAMES = [
    "Iyer", "Nair", "Reddy", "Menon", "Sharma", "Patel", "Kumar", "Das", "Singh", "Rao",
    "Pillai", "Gupta", "Joshi", "Khan", "Mehta", "Bose", "Chandra", "Naidu", "Verma", "Shetty",
]
DEPARTMENTS = [
    "Artificial Intelligence and Data Science", "Computer Science",
    "Information Technology", "Electronics",
]


def make_students(rng: np.random.Generator) -> pd.DataFrame:
    n = N_STUDENTS
    ability = rng.normal(0, 1, n)
    engagement = 0.5 * ability + rng.normal(0, 0.9, n)

    attendance = np.clip(80 + 8 * engagement + rng.normal(0, 5, n), 35, 100)
    internal = np.clip(62 + 13 * ability + 0.2 * (attendance - 80) + rng.normal(0, 8, n), 8, 100)
    assignment = np.clip(68 + 10 * ability + 5 * engagement + rng.normal(0, 9, n), 5, 100)
    previous = np.clip(63 + 12 * ability + rng.normal(0, 8, n), 10, 100)
    subjects = {
        "statistics_marks": np.clip(internal + rng.normal(0, 7, n), 5, 100),
        "programming_marks": np.clip(internal + rng.normal(1, 8, n), 5, 100),
        "communication_marks": np.clip(internal + rng.normal(3, 9, n), 5, 100),
    }

    # Artificial outcome: logistic chance of failing, with random noise included.
    risk = (
        0.075 * (62 - internal) + 0.055 * (76 - attendance)
        + 0.04 * (63 - assignment) + 0.035 * (60 - previous)
        + rng.normal(0, 0.8, n) - 1.2
    )
    p_fail = 1 / (1 + np.exp(-risk))
    failed = rng.random(n) < p_fail

    frame = pd.DataFrame({
        "student_id": [f"STU{i:04d}" for i in range(1, n + 1)],
        "student_name": [
            f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)
        ],
        "department": rng.choice(DEPARTMENTS, n, p=[0.3, 0.3, 0.25, 0.15]),
        "year": rng.choice([1, 2, 3], n, p=[0.35, 0.35, 0.3]),
        "section": rng.choice(["A", "B", "C"], n),
        "attendance_percentage": attendance.round(1),
        "internal_marks": internal.round(0),
        "assignment_score": assignment.round(0),
        "previous_semester_marks": previous.round(0),
        **{k: v.round(0) for k, v in subjects.items()},
        "final_result": np.where(failed, "Fail", "Pass"),
    })
    return frame


def add_quality_issues(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frame = frame.copy()
    n = len(frame)
    pick = lambda k: rng.choice(n, k, replace=False)  # noqa: E731

    # Students with no outcome yet (current cohort) - scored by the model.
    unlabelled = pick(12)
    frame["final_result"] = frame["final_result"].astype(object)
    frame.loc[unlabelled, "final_result"] = np.nan

    # Missing values
    frame["assignment_score"] = frame["assignment_score"].astype(object)
    frame.loc[pick(6), "assignment_score"] = np.nan
    frame.loc[pick(4), "attendance_percentage"] = np.nan
    frame.loc[pick(3), "previous_semester_marks"] = np.nan
    frame.loc[pick(3), "programming_marks"] = np.nan

    # Invalid values
    frame.loc[pick(1), "attendance_percentage"] = 104.5      # above 100
    frame.loc[pick(1), "internal_marks"] = -6                # below 0
    frame.loc[pick(1), "communication_marks"] = 112          # above 100
    frame.loc[pick(1), "assignment_score"] = "absent"        # not a number

    # Inconsistent text
    frame.loc[pick(4), "final_result"] = "pass"
    frame.loc[pick(3), "final_result"] = "FAIL"
    spaced = pick(3)
    frame.loc[spaced, "department"] = frame.loc[spaced, "department"].astype(str) + " "
    frame.loc[pick(2), "section"] = " B"

    # Repeated student ID with different values (kept and flagged by the app)
    frame.loc[n - 1, "student_id"] = frame.loc[10, "student_id"]

    # Exact duplicate rows appended at the end
    duplicates = frame.iloc[[5, 42, 117]].copy()
    return pd.concat([frame, duplicates], ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    frame = add_quality_issues(make_students(rng), rng)
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame)} rows to {OUTPUT}")
    outcome = frame["final_result"].astype(str).str.strip().str.lower()
    print(outcome.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
