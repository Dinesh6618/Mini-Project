"""Unit tests for the analysis modules.

Run from the project folder:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from modules import database, exploratory_analysis as eda  # noqa: E402
from modules.data_cleaning import clean_dataset, normalize_outcome  # noqa: E402
from modules.data_loader import (  # noqa: E402
    DataValidationError, anonymize_frame, load_csv, read_csv_source,
)
from modules.prediction import (  # noqa: E402
    ModelTrainingError, apply_risk_categories, explain_single, predict_single,
    risk_category, score_students, train_models,
)
from modules.recommendations import academic_recommendations, needs_follow_up  # noqa: E402

SAMPLE = ROOT / "data" / "sample_students.csv"


def make_frame(n: int = 120, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "student_id": [f"S{i}" for i in range(n)],
        "attendance_percentage": np.clip(78 + 8 * ability + rng.normal(0, 5, n), 30, 100),
        "internal_marks": np.clip(60 + 14 * ability + rng.normal(0, 6, n), 5, 100),
        "assignment_score": np.clip(65 + 10 * ability + rng.normal(0, 8, n), 5, 100),
        "previous_semester_marks": np.clip(62 + 12 * ability + rng.normal(0, 7, n), 5, 100),
    })
    if with_target:
        p = 1 / (1 + np.exp(-(-1.0 - 1.3 * ability + rng.normal(0, 0.6, n))))
        frame["final_result"] = np.where(rng.random(n) < p, "Fail", "Pass")
    return frame


class LoaderTests(unittest.TestCase):
    def test_empty_bytes(self):
        with self.assertRaises(DataValidationError):
            read_csv_source(b"   \n")

    def test_header_only(self):
        with self.assertRaises(DataValidationError) as ctx:
            load_csv(b"student_id,attendance_percentage,internal_marks,assignment_score,previous_semester_marks\n")
        self.assertIn("no student records", str(ctx.exception))

    def test_missing_columns_named(self):
        with self.assertRaises(DataValidationError) as ctx:
            load_csv(b"student_id,internal_marks\nA,50\n")
        message = str(ctx.exception)
        self.assertIn("attendance_percentage", message)
        self.assertIn("assignment_score", message)

    def test_duplicate_column_names(self):
        with self.assertRaises(DataValidationError):
            load_csv(b"student_id,Student_ID,internal_marks\nA,B,50\n")

    def test_column_names_standardised(self):
        text = (
            " Student ID ,Attendance_Percentage,INTERNAL_MARKS,assignment_score,previous_semester_marks\n"
            "A,90,50,60,55\n"
        )
        # "Student ID" -> student_id
        frame = load_csv(text.encode())
        self.assertIn("student_id", frame.columns)
        self.assertIn("internal_marks", frame.columns)

    def test_latin1_file(self):
        text = ("student_id,student_name,attendance_percentage,internal_marks,"
                "assignment_score,previous_semester_marks\nA,Jos\xe9,90,50,60,55\n")
        frame = load_csv(text.encode("latin-1"))
        self.assertEqual(frame.loc[0, "student_name"], "Jos\xe9")

    def test_non_numeric_required_column(self):
        text = ("student_id,attendance_percentage,internal_marks,assignment_score,"
                "previous_semester_marks\nA,high,50,60,55\nB,low,40,50,45\n")
        with self.assertRaises(DataValidationError):
            load_csv(text.encode())

    def test_sample_loads(self):
        frame = load_csv(SAMPLE)
        self.assertGreater(len(frame), 200)

    def test_anonymize(self):
        frame = pd.DataFrame({"student_id": ["a", "b"], "student_name": ["x", "y"], "v": [1, 2]})
        result = anonymize_frame(frame)
        self.assertEqual(result["student_id"].tolist(), ["ANON-0001", "ANON-0002"])
        self.assertNotIn("student_name", result.columns)
        self.assertIn("student_name", frame.columns)  # original untouched


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "student_id": ["A", "B", "C", "C", "D", "A"],
            "department": [" CS ", "CS", "IT", "IT", "CS", " CS "],
            "year": [1, 1, 2, 2, "2", 1],
            "attendance_percentage": [90, 120, 80, 80, "n/a", 90],
            "internal_marks": [50, 60, -5, -5, 70, 50],
            "assignment_score": [60, np.nan, 70, 70, 65, 60],
            "previous_semester_marks": [55, 65, 70, 70, 60, 55],
            "final_result": ["pass", "FAIL", "Passed", "Passed", None, "pass"],
        })

    def test_invalid_values_become_missing_and_are_logged(self):
        cleaned, report = clean_dataset(self.frame)
        self.assertEqual(report["issues"].set_index("Column").loc["attendance_percentage", "Outside range"], 1)
        self.assertEqual(report["issues"].set_index("Column").loc["attendance_percentage", "Not numeric"], 1)
        log = report["change_log"]
        self.assertTrue((log["Column"] == "attendance_percentage").sum() >= 2)
        self.assertIn("120", [str(v) for v in log["Original value"]])
        self.assertTrue(cleaned["attendance_percentage"].isna().sum() >= 2)

    def test_duplicates_removed_once(self):
        cleaned, report = clean_dataset(self.frame)
        self.assertEqual(report["duplicates_removed"], 2)
        self.assertEqual(len(cleaned), len(self.frame) - 2)
        self.assertEqual(report["rows_output"], len(cleaned))

    def test_label_normalisation(self):
        cleaned, _ = clean_dataset(self.frame)
        self.assertEqual(set(cleaned["final_result"].dropna()), {"Pass", "Fail"})
        self.assertEqual(normalize_outcome("  pAsSeD "), "Pass")
        self.assertIsNone(normalize_outcome("maybe"))

    def test_no_imputation_and_input_unchanged(self):
        before = self.frame.copy()
        cleaned, _ = clean_dataset(self.frame)
        pd.testing.assert_frame_equal(self.frame, before)
        self.assertTrue(cleaned["assignment_score"].isna().any())

    def test_whitespace_trimmed(self):
        cleaned, _ = clean_dataset(self.frame)
        self.assertNotIn(" CS ", cleaned["department"].tolist())

    def test_every_change_is_itemised_in_the_log(self):
        _, report = clean_dataset(self.frame)
        log = report["change_log"]
        actions = " | ".join(log["Action"])
        self.assertIn("Spaces trimmed", actions)             # ' CS ' values
        self.assertIn("Label standardised", actions)         # 'pass', 'FAIL', 'Passed'
        self.assertIn("exact duplicate", actions)
        self.assertIn("Outside 0-100", actions)
        self.assertIn("Not a number", actions)

    def test_idempotent(self):
        once, _ = clean_dataset(self.frame)
        twice, report = clean_dataset(once)
        self.assertEqual(len(once), len(twice))
        self.assertEqual(report["duplicates_removed"], 0)
        self.assertTrue(report["change_log"].empty)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = make_frame()
        cls.trained = train_models(cls.frame)

    def test_both_models_and_metrics(self):
        self.assertEqual(set(self.trained["models"]), {"Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors"})
        for result in self.trained["results"].values():
            for key in ("accuracy", "precision", "recall", "f1"):
                self.assertTrue(0.0 <= result["metrics"][key] <= 1.0)
            self.assertEqual(int(result["confusion_matrix"].sum()), self.trained["n_test"])

    def test_no_leakage_columns(self):
        features = set(self.trained["features"])
        self.assertTrue(features.isdisjoint({"final_result", "student_id", "student_name",
                                             "department", "year", "section"}))

    def test_metrics_match_recomputation(self):
        from sklearn.metrics import f1_score
        result = self.trained["results"]["Logistic Regression"]
        self.assertAlmostEqual(
            result["metrics"]["f1"], f1_score(result["y_true"], result["y_pred"]), places=9
        )

    def test_split_sizes(self):
        self.assertEqual(self.trained["n_train"] + self.trained["n_test"], self.trained["n_total"])

    def test_deterministic(self):
        again = train_models(self.frame)
        for name in self.trained["results"]:
            self.assertEqual(self.trained["results"][name]["metrics"], again["results"][name]["metrics"])

    def test_too_small(self):
        with self.assertRaises(ModelTrainingError):
            train_models(make_frame(n=20))

    def test_no_target_column(self):
        with self.assertRaises(ModelTrainingError) as ctx:
            train_models(make_frame(with_target=False))
        self.assertIn("final_result", str(ctx.exception))

    def test_single_class(self):
        frame = make_frame()
        frame["final_result"] = "Pass"
        with self.assertRaises(ModelTrainingError):
            train_models(frame)

    def test_cross_validation_only_when_large(self):
        self.assertTrue(self.trained["cross_validation"])  # 120 rows -> 90 training rows
        small = train_models(make_frame(n=60, seed=3))
        self.assertEqual(small["cross_validation"], [])
        self.assertTrue(small["cv_note"])

    def test_risk_category_thresholds(self):
        self.assertEqual(risk_category(0.10, 0.35, 0.65), "Low Risk")
        self.assertEqual(risk_category(0.35, 0.35, 0.65), "Moderate Risk")
        self.assertEqual(risk_category(0.65, 0.35, 0.65), "High Risk")
        self.assertEqual(risk_category(float("nan"), 0.35, 0.65), "Not scored")

    def test_scoring_and_single_prediction(self):
        frame = make_frame()
        frame.loc[:9, "final_result"] = np.nan  # current cohort without outcomes
        scores = score_students(frame, self.trained, "Logistic Regression")
        self.assertEqual(len(scores), len(frame))
        self.assertTrue(scores.between(0, 1).all())
        categories = apply_risk_categories(scores, 0.35, 0.65)
        self.assertTrue(set(categories) <= {"Low Risk", "Moderate Risk", "High Risk"})

        strong = {"attendance_percentage": 95, "internal_marks": 90,
                  "assignment_score": 90, "previous_semester_marks": 88}
        weak = {"attendance_percentage": 45, "internal_marks": 25,
                "assignment_score": 30, "previous_semester_marks": 35}
        for name in self.trained["models"]:
            self.assertLess(predict_single(self.trained, name, strong),
                            predict_single(self.trained, name, weak))
        contributions = explain_single(self.trained, "Logistic Regression", weak)
        self.assertEqual(set(contributions["Feature"]), set(self.trained["features"]))
        self.assertGreater(contributions["Contribution"].sum(), 0)
        self.assertIsNone(explain_single(self.trained, "Decision Tree", weak))

    def test_sample_dataset_trains(self):
        cleaned, _ = clean_dataset(load_csv(SAMPLE))
        trained = train_models(cleaned)
        self.assertGreater(trained["n_total"], 200)
        self.assertEqual(trained["n_unlabelled"], 12)


class RecommendationTests(unittest.TestCase):
    def test_supportive_suggestions(self):
        tips = academic_recommendations({
            "attendance_percentage": 60, "internal_marks": 30, "assignment_score": 40,
            "previous_semester_marks": 60,
        })
        self.assertEqual(len(tips), 3 + 0 + 1)  # attendance, internal, assignment, decline
        joined = " ".join(tips).lower()
        for punitive in ("penal", "punish", "fail the student", "warning letter"):
            self.assertNotIn(punitive, joined)

    def test_no_suggestions_when_fine(self):
        self.assertEqual(academic_recommendations({
            "attendance_percentage": 90, "internal_marks": 70, "assignment_score": 70,
            "previous_semester_marks": 68,
        }), [])

    def test_missing_values_skipped(self):
        self.assertEqual(academic_recommendations({"attendance_percentage": float("nan")}), [])

    def test_custom_thresholds(self):
        tips = academic_recommendations({"attendance_percentage": 80}, {"attendance_percentage": 85})
        self.assertEqual(len(tips), 1)

    def test_follow_up_mask(self):
        frame = pd.DataFrame({"attendance_percentage": [90, 60], "internal_marks": [70, 70]})
        self.assertEqual(needs_follow_up(frame).tolist(), [False, True])


class DatabaseTests(unittest.TestCase):
    def test_round_trip_drops_names(self):
        frame = pd.DataFrame({"student_id": ["a"], "student_name": ["n"], "internal_marks": [50.0]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.db"
            self.assertFalse(database.database_exists(path))
            self.assertEqual(database.save_dataset(frame, path), 1)
            self.assertTrue(database.database_exists(path))
            loaded = database.load_dataset(path)
            self.assertNotIn("student_name", loaded.columns)
            self.assertEqual(loaded.loc[0, "internal_marks"], 50.0)
            database.save_dataset(frame, path, include_names=True)
            self.assertIn("student_name", database.load_dataset(path).columns)
            database.delete_database(path)
            self.assertFalse(path.exists())


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = clean_dataset(load_csv(SAMPLE))

    def test_charts_build(self):
        f = self.frame
        self.assertIsNotNone(eda.histogram(f, "internal_marks"))
        self.assertIsNotNone(eda.performance_band_chart(f))
        self.assertIsNotNone(eda.subject_bar_chart(f))
        self.assertIsNotNone(eda.subject_box_chart(f))
        self.assertIsNotNone(eda.change_histogram(f))
        self.assertIsNotNone(eda.assignment_by_outcome(f))
        self.assertIsNotNone(eda.performance_trend_by_group(f, "year"))
        fig, r, n = eda.previous_vs_current(f)
        self.assertTrue(-1 <= r <= 1)
        self.assertGreater(n, 100)
        self.assertIsNotNone(eda.correlation_heatmap(eda.correlation_matrix(f)))

    def test_missing_columns_return_none(self):
        bare = pd.DataFrame({"x": [1, 2, 3]})
        self.assertIsNone(eda.histogram(bare, "internal_marks"))
        self.assertIsNone(eda.subject_bar_chart(bare))
        self.assertIsNone(eda.previous_vs_current(bare))
        self.assertTrue(eda.find_outliers(bare).empty)

    def test_outliers_use_iqr_rule(self):
        values = [50.0] * 20 + [51.0] * 20 + [5.0]
        frame = pd.DataFrame({"student_id": [f"s{i}" for i in range(41)], "internal_marks": values})
        outliers = eda.find_outliers(frame)
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers.iloc[0]["Student ID"], "s40")

    def test_correlation_description(self):
        self.assertEqual(eda.describe_correlation(0.9), "very strong positive")
        self.assertEqual(eda.describe_correlation(-0.1), "very weak negative")


if __name__ == "__main__":
    unittest.main()
