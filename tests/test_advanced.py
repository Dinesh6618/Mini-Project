"""Advanced calculations and real Streamlit control integration."""
import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from modules.advanced import quality_assessment, semester_progress, class_comparison, fairness_audit
from modules.prediction import train_models, predict_single, sensitivity_explanation, explain_single, score_students, ModelTrainingError
from modules.data_loader import anonymize_frame
from modules.data_cleaning import clean_dataset
from modules.reporting import analysis_pdf
from tests.test_modules import make_frame
from tests.test_app import new_app, app_with_csv


class AdvancedCalculations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = make_frame(n=120)
        cls.trained = train_models(cls.frame)

    def test_quality_formula_and_detected_counts(self):
        frame = make_frame(n=4)
        frame.loc[0, "internal_marks"] = np.nan
        frame.loc[1, "assignment_score"] = 150
        frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
        score, components = quality_assessment(frame)
        self.assertEqual(components.Issues.tolist(), [1, 1, 1, 0])
        self.assertAlmostEqual(score, 100 * (1 - (1/frame.size + 1/frame.size + 1/5)/4))
        self.assertEqual(quality_assessment(make_frame(n=4))[0], 100)

    def test_conflicting_records_and_legitimate_semesters(self):
        frame = make_frame(n=2)
        frame.student_id = "same"
        frame["semester"] = [1, 2]
        self.assertEqual(quality_assessment(frame)[1].Issues.iloc[-1], 0)
        frame.semester = 1
        self.assertEqual(quality_assessment(frame)[1].Issues.iloc[-1], 2)

    def test_progress_order_and_actual_differences(self):
        frame = make_frame(n=3)
        frame.student_id = "student"
        frame["semester"] = [3, 1, 2]
        frame.internal_marks = [65, 50, 70]
        result = semester_progress(frame)
        self.assertEqual(result.semester.tolist(), [1, 2, 3])
        self.assertTrue(pd.isna(result.internal_marks_change.iloc[0]))
        self.assertEqual(result.internal_marks_change.iloc[1:].tolist(), [20, -5])
        anonymous = anonymize_frame(frame)
        self.assertEqual(anonymous.student_id.nunique(), 1)
        self.assertEqual(len(semester_progress(anonymous)), 3)
        self.assertEqual(clean_dataset(frame)[1]["duplicate_ids"], 0)

    def test_progress_rejects_ambiguous_or_conflicting_history(self):
        frame = make_frame(n=3)
        with self.assertRaises(ValueError):
            semester_progress(frame)
        frame["semester"] = [1, "autumn", 3]
        with self.assertRaises(ValueError):
            semester_progress(frame)
        frame.semester = 1
        frame.student_id = "same"
        with self.assertRaises(ValueError):
            semester_progress(frame)

    def test_group_aggregates_with_missing_values(self):
        frame = pd.DataFrame({"section": ["A", "A", "B"], "math_marks": [20, np.nan, 80]})
        means, counts = class_comparison(frame, "section", ["math_marks"])
        self.assertEqual(means.loc["A", "math_marks"], 20)
        self.assertEqual(counts.loc["A", "math_marks"], 1)
        self.assertEqual(means.loc["B", "math_marks"], 80)

    def test_scaling_uses_only_training_records(self):
        pipeline = self.trained["models"]["Logistic Regression"]
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        features = self.trained["features"]
        train = self.frame.loc[self.trained["train_index"], features]
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, train.median().to_numpy())
        np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, train.fillna(train.median()).mean().to_numpy())

    def test_four_models_and_sensitivity_are_calculated(self):
        values = self.frame.iloc[0].to_dict()
        for name in self.trained["models"]:
            explanation = sensitivity_explanation(self.trained, name, values)
            row = explanation.iloc[0]
            reference = dict(values, **{row.Feature: row["Training reference"]})
            self.assertAlmostEqual(row["Score difference"], predict_single(self.trained, name, values) - predict_single(self.trained, name, reference))
        local = explain_single(self.trained, "Logistic Regression", values)
        intercept = self.trained["models"]["Logistic Regression"].named_steps["model"].intercept_[0]
        probability = 1 / (1 + np.exp(-(intercept + local.Contribution.sum())))
        self.assertAlmostEqual(probability, predict_single(self.trained, "Logistic Regression", values))

    def test_repeated_students_never_cross_holdout(self):
        frame = pd.concat([self.frame.assign(semester=1), self.frame.assign(semester=2)], ignore_index=True)
        trained = train_models(frame)
        self.assertTrue(trained["grouped_split"])
        train_ids = set(frame.loc[trained["train_index"], "student_id"])
        test_ids = set(frame.loc[trained["test_index"], "student_id"])
        self.assertFalse(train_ids & test_ids)
        self.assertTrue(score_students(frame, trained, "K-Nearest Neighbors").notna().all())

    def test_too_few_independent_students_refuses_training(self):
        frame = pd.concat([self.frame.iloc[:6]] * 10, ignore_index=True)
        with self.assertRaises(ModelTrainingError):
            train_models(frame)

    def test_empty_academic_feature_retains_explanation_shape(self):
        frame = self.frame.copy()
        frame["empty_marks"] = np.nan
        trained = train_models(frame)
        self.assertEqual(len(explain_single(trained, "Logistic Regression", frame.iloc[0].to_dict())), len(trained["features"]))

    def test_fairness_uses_held_out_errors_and_suppresses_small_groups(self):
        frame = pd.DataFrame({"student_id": [f"S{i}" for i in range(43)], "section": ["A"]*20 + ["B"]*20 + ["SECRET"]*3})
        true = np.array([0]*10 + [1]*10 + [0]*10 + [1]*10 + [0]*3)
        pred = true.copy()
        pred[0] = 1
        pred[10] = 0
        trained = {"test_index": frame.index.tolist(), "results": {"test": {"y_true": true, "y_pred": pred}}}
        result = fairness_audit(frame, trained, "test", "section")
        self.assertEqual(result.Group.tolist(), ["A", "B"])
        self.assertAlmostEqual(result.iloc[0]["Error rate"], .1)
        self.assertAlmostEqual(result.iloc[0]["False positive rate"], .1)
        self.assertAlmostEqual(result.iloc[0]["False negative rate"], .1)
        with self.assertRaises(ValueError):
            fairness_audit(frame, trained, "test", "gender")

    def test_pdf_is_real_document_with_and_without_labels(self):
        cleaned, report = clean_dataset(self.frame)
        ctx = SimpleNamespace(raw=self.frame, cleaned=cleaned, view=anonymize_frame(cleaned).iloc[:5], report=report, source_label="Test fixture", model_name="Logistic Regression", moderate=.5, high=.75, anonymize=True, selected_filters={"section": ["A"]}, has_model=True, trained=self.trained, using_sample=True)
        content = analysis_pdf(ctx)
        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertTrue(content.rstrip().endswith(b"%%EOF"))
        self.assertGreater(content.count(b"/Type /Page"), 4)
        ctx.has_model = False
        ctx.training_error = "No labels"
        ctx.view = ctx.view.iloc[:0]
        self.assertTrue(analysis_pdf(ctx).startswith(b"%PDF-"))


class AdvancedControls(unittest.TestCase):
    def test_simulator_reacts_and_keeps_baseline_fixed(self):
        at = new_app()
        at.sidebar.radio(key="page").set_value("What-If Simulator").run()
        baseline = next(m.value for m in at.metric if m.label == "Baseline estimate")
        at.sidebar.selectbox(key="risk_model").set_value("Logistic Regression").run()
        sliders = [w for w in at.slider if w.key.startswith("scenario_")]
        for slider in sliders:
            slider.set_value(5.0)
        at.run()
        low = next(m.value for m in at.metric if m.label == "Simulated estimate")
        fixed = next(m.value for m in at.metric if m.label == "Baseline estimate")
        for slider in [w for w in at.slider if w.key.startswith("scenario_")]:
            slider.set_value(95.0)
        at.run()
        high = next(m.value for m in at.metric if m.label == "Simulated estimate")
        self.assertNotEqual(low, high)
        self.assertEqual(fixed, next(m.value for m in at.metric if m.label == "Baseline estimate"))
        self.assertFalse(at.exception)

    def test_report_and_comparison_handle_empty_filters(self):
        at = new_app()
        department = next(w for w in at.sidebar.multiselect if w.label == "Department")
        department.set_value([]).run()
        for page in ["What-If Simulator", "Report Generator", "Class and Subject Comparison", "Semester Progress Tracker"]:
            at.sidebar.radio(key="page").set_value(page).run()
            self.assertFalse(at.exception, page)
            self.assertFalse([e for e in at.error if "could not be displayed" in str(e.value)], page)

    def test_missing_baseline_slider_state_persists(self):
        at = new_app()
        at.sidebar.radio(key="page").set_value("What-If Simulator").run()
        import modules.data_loader as loader
        from modules.data_cleaning import clean_dataset
        from tests.test_app import ROOT
        cleaned, _ = clean_dataset(loader.load_csv(ROOT / "data" / "sample_students.csv"))
        missing = cleaned[cleaned.attendance_percentage.isna()].index[0]
        next(w for w in at.selectbox if w.label == "Baseline student record").set_value(missing).run()
        widget = next(w for w in at.slider if w.key.startswith(f"scenario_{missing}_attendance_percentage"))
        key = widget.key
        widget.set_value(42.0).run()
        self.assertEqual(at.slider(key=key).value, 42.0)
        self.assertFalse(at.exception)

    def test_history_page_uses_shared_anonymous_identity(self):
        import tempfile
        from pathlib import Path
        frame = pd.concat([make_frame(n=60).assign(semester=1), make_frame(n=60).assign(semester=2)], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            frame.to_csv(path, index=False)
            at = app_with_csv(path)
            at.sidebar.radio(key="page").set_value("Semester Progress Tracker").run()
            self.assertFalse(at.exception)
            self.assertFalse([e for e in at.error if "could not be displayed" in str(e.value)])
            self.assertEqual(len(next(w for w in at.selectbox if w.label == "Student").options), 60)


if __name__ == "__main__":
    unittest.main()
