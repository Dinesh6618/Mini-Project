"""Headless end-to-end tests: run the real Streamlit script and visit every page.

Run from the project folder:  python -m unittest tests.test_app -v
"""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path

from streamlit.testing.v1 import AppTest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "Dashboard", "Student Records", "Data Cleaning", "Exploratory Analysis",
    "Risk Prediction", "Model Evaluation", "Data Ethics", "About Project",
    "What-If Simulator", "Semester Progress Tracker", "Model Comparison Lab",
    "Report Generator", "Data Quality Score", "Class and Subject Comparison", "Bias and Fairness Auditor",
]


def new_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


class AppTests(unittest.TestCase):
    def test_starts_on_dashboard_without_errors(self):
        at = new_app()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(at.title[0].value, "Academic Performance Dashboard")
        labels = [m.label for m in at.metric]
        self.assertIn("Total students", labels)
        self.assertIn("Average internal marks", labels)
        self.assertIn("Average attendance", labels)

    def test_every_page_renders(self):
        at = new_app()
        for page in PAGES:
            at.sidebar.radio(key="page").set_value(page).run()
            self.assertEqual(len(at.exception), 0, f"{page}: {[e.value for e in at.exception]}")
            self.assertFalse(
                [e for e in at.error if "could not be displayed" in str(e.value)],
                f"{page} raised a handled error",
            )

    def test_dashboard_shortcuts_open_the_correct_pages(self):
        at = new_app()
        for destination in ["Student Records", "What-If Simulator", "Report Generator"]:
            at.sidebar.radio(key="page").set_value("Dashboard").run()
            at.button(key=f"go_{destination}").click().run()
            self.assertEqual(at.session_state["page"], destination)
            self.assertFalse(at.exception)

    def test_metrics_are_computed_from_data(self):
        import pandas as pd
        from modules.data_cleaning import clean_dataset
        from modules.data_loader import load_csv
        cleaned, _ = clean_dataset(load_csv(ROOT / "data" / "sample_students.csv"))
        at = new_app()
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Total students"], f"{len(cleaned):,}")
        self.assertEqual(metrics["Average internal marks"], f"{cleaned['internal_marks'].mean():.1f} / 100")
        self.assertEqual(metrics["Average attendance"], f"{cleaned['attendance_percentage'].mean():.1f}%")
        self.assertIsInstance(pd.Series([1]), pd.Series)

    def test_filter_changes_dashboard(self):
        at = new_app()
        total = next(m.value for m in at.metric if m.label == "Total students")
        department = next(w for w in at.sidebar.multiselect if w.label == "Department")
        department.set_value(department.options[:1]).run()
        self.assertEqual(len(at.exception), 0)
        filtered = next(m.value for m in at.metric if m.label == "Total students")
        self.assertNotEqual(total, filtered)
        self.assertLess(int(filtered.replace(",", "")), int(total.replace(",", "")))

    def test_threshold_change_updates_support_count(self):
        at = new_app()
        before = next(m.value for m in at.metric if m.label == "May need support")
        at.sidebar.slider(key="moderate").set_value(0.70).run()
        after = next(m.value for m in at.metric if m.label == "May need support")
        self.assertNotEqual(before, after)

    def test_anonymous_ids_toggle(self):
        at = new_app()
        at.sidebar.radio(key="page").set_value("Student Records").run()
        self.assertTrue(any("Anonymous IDs are on" in str(i.value) for i in at.info))
        at.sidebar.checkbox(key="anonymize").set_value(False).run()
        self.assertTrue(any("Anonymous IDs are off" in str(w.value) for w in at.warning))

    def test_upload_without_file_falls_back_to_labelled_sample(self):
        at = new_app()
        at.sidebar.radio(key="source").set_value("Upload a CSV file").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("No file uploaded yet" in str(i.value) for i in at.sidebar.info))
        # The page still works and clearly says which data it is showing.
        self.assertEqual(at.title[0].value, "Academic Performance Dashboard")
        self.assertTrue(any("no file uploaded yet" in str(m.value) for m in at.markdown))

    def test_prediction_form_submits(self):
        at = new_app()
        at.sidebar.radio(key="page").set_value("Risk Prediction").run()
        at.number_input(key="input_attendance_percentage").set_value(40.0)
        at.number_input(key="input_internal_marks").set_value(20.0)
        at.number_input(key="input_assignment_score").set_value(25.0)
        at.number_input(key="input_previous_semester_marks").set_value(30.0)
        next(b for b in at.button if b.label == "Estimate risk").click().run()
        self.assertEqual(len(at.exception), 0)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Estimated category"], "High Risk")
        self.assertTrue(metrics["Estimated non-pass probability"].endswith("%"))


def app_with_csv(csv_path: Path) -> AppTest:
    """Run the real app but make the 'sample dataset' load ``csv_path`` instead."""
    code = f"""
import sys, runpy
sys.path.insert(0, r"{ROOT}")
import modules.data_loader as loader
_real = loader.load_csv
loader.load_csv = lambda source, _r=_real: _r(r"{csv_path}")
try:
    runpy.run_path(r"{ROOT / 'app.py'}", run_name="__main__")
finally:
    loader.load_csv = _real
"""
    return AppTest.from_string(code, default_timeout=120).run()


class UnhappyPathTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, frame) -> Path:
        path = Path(self.folder.name) / "data.csv"
        frame.to_csv(path, index=False)
        return path

    def visit_all(self, at: AppTest) -> None:
        for page in PAGES:
            at.sidebar.radio(key="page").set_value(page).run()
            self.assertEqual(len(at.exception), 0, f"{page}: {[e.value for e in at.exception]}")
            self.assertFalse(
                [e for e in at.error if "could not be displayed" in str(e.value)], page
            )

    def test_dataset_without_outcome_column(self):
        from tests.test_modules import make_frame
        frame = make_frame(n=80, with_target=False)
        at = app_with_csv(self.write(frame))
        self.assertEqual(len(at.exception), 0)
        labels = [m.label for m in at.metric]
        self.assertIn("Below an indicator", labels)  # indicator fallback, no model
        self.visit_all(at)
        at.sidebar.radio(key="page").set_value("Risk Prediction").run()
        self.assertTrue(any("final_result" in str(w.value) for w in at.warning))
        self.assertFalse(at.button, "no prediction form should exist without a valid model")
        at.sidebar.radio(key="page").set_value("Model Evaluation").run()
        self.assertFalse(at.metric, "no metrics may be shown without a trained model")

    def test_dataset_too_small_to_train(self):
        from tests.test_modules import make_frame
        at = app_with_csv(self.write(make_frame(n=25)))
        self.visit_all(at)
        at.sidebar.radio(key="page").set_value("Risk Prediction").run()
        self.assertTrue(any("labelled records" in str(w.value) for w in at.warning))

    def test_every_value_missing_in_a_column_is_rejected_at_load(self):
        from modules.data_loader import DataValidationError, load_csv
        from tests.test_modules import make_frame
        frame = make_frame(n=30)
        frame["assignment_score"] = "n/a"
        with self.assertRaises(DataValidationError):
            load_csv(self.write(frame))

    def test_two_row_dataset_does_not_crash(self):
        from tests.test_modules import make_frame
        at = app_with_csv(self.write(make_frame(n=2)))
        self.visit_all(at)


if __name__ == "__main__":
    unittest.main()
