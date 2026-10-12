# Student Academic Performance Analysis and Early Risk Prediction

A local Streamlit application that analyses student academic records, shows patterns, and uses four machine-learning models with transparent explanations to estimate which students may benefit from extra academic support.

It is an **educational decision-support tool**. It does not label, rank or penalise students, and no estimate should ever be the only basis for an academic decision.

Mini project for **Principles of Data Science**. Python, pandas, NumPy, scikit-learn, Plotly, Matplotlib, Streamlit and optional SQLite. No paid APIs or cloud services; everything runs on your own computer.

---

## 1. Run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, if `streamlit` is not recognised, use `python -m streamlit run app.py`.

The app opens on the dashboard using the built-in **synthetic sample dataset**. Use the sidebar to upload your own CSV.

Run the automated tests (about 30 seconds):

```bash
python -m unittest discover -s tests -t .
```

## 2. What each page does

| Page | What it shows | Main concept |
|---|---|---|
| **Dashboard** | Total students, average internal marks, average attendance, students who may need support, performance distribution, attendance vs marks, subject comparison, trends, risk categories. Updates with the sidebar filters. | Descriptive statistics, visualisation |
| **Student Records** | Searchable table, filters (department, year, section, risk category), CSV downloads, per-student support suggestions, optional SQLite save. | Data management, privacy |
| **Data Cleaning** | Quality report, warnings, operations performed, and an itemised change log with the original value of every change. | Data preprocessing |
| **Exploratory Analysis** | Distributions, correlations (with a causation warning), assignment analysis, previous vs current marks, subject analysis, IQR outliers. | EDA |
| **Risk Prediction** | Enter a student's indicators, get a Low / Moderate / High estimate with the factors behind it and supportive suggestions. Thresholds are configurable. | Supervised learning |
| **Model Evaluation** | Held-out test metrics, confusion matrix, coefficients / importances, cross-validation, limitations. | Model evaluation |
| **Data Ethics** | Privacy, appropriate use, bias, transparency, limitations, human oversight. | Ethics |
| **About Project** | Summary, syllabus concept map, sample-data description, module guide. | Documentation |

Sidebar controls: dataset source, anonymous IDs (on by default), filters, and **Risk settings** (model choice, risk thresholds, indicator thresholds).

## 3. Project structure

```
app.py                         Page setup, sidebar, data loading, caching, routing
requirements.txt
README.md
.streamlit/config.toml         Dark navy / teal theme
data/
  sample_students.csv          Fictional sample dataset (300 students + 3 deliberate duplicate rows)
  generate_sample_data.py      Script that generated it (fixed seed)
modules/                       Data-science logic (no Streamlit layout code)
  config.py                    Column names, valid ranges, thresholds, target definition
  data_loader.py               CSV reading, column standardising, validation, anonymising
  data_cleaning.py             Type conversion, range checks, duplicates, change log
  exploratory_analysis.py      Statistics, correlations, outliers, Plotly charts
  prediction.py                Pipelines, Logistic Regression, Decision Tree, risk scoring
  model_evaluation.py          Metric tables, confusion matrix, importance charts, warnings
  recommendations.py           Rule-based, supportive suggestions
  database.py                  Optional local SQLite storage
  theme.py                     Colours, CSS, shared chart styling
views/                         One file per page; each has render(ctx)
tests/                         Unit tests and headless end-to-end tests
```

## 4. Input data

Required columns:

`student_id, attendance_percentage, internal_marks, assignment_score, previous_semester_marks`

Optional columns: `student_name, department, year, section, final_result` and any number of subject columns ending in `_marks` (for example `statistics_marks`). Attendance and scores use a 0-100 scale. Column names are trimmed and lower-cased automatically.

### Validation and error messages

| Problem | What happens |
|---|---|
| Empty file, or header only | Clear error; nothing is analysed |
| Missing required column(s) | Error naming the missing columns and the expected ones |
| Repeated column names | Error naming them |
| A required column has no numeric values | Error naming the column |
| Non-numeric entries, values outside 0-100 | Set to missing, counted, and listed in the change log |
| Exact duplicate rows | First kept, others removed, listed in the change log |
| Repeated `student_id` with different values | Kept, flagged for manual review (never merged) |
| Extra spaces, `pass` / `FAIL` / `Passed` labels | Standardised, each change listed |
| Missing values | **Not imputed** for analysis. The model pipeline fills them with medians learned from the training split only |

Nothing is modified silently: every change appears under **Data Cleaning**.

## 5. Sample dataset

`data/sample_students.csv` is **fictional and synthetic**. No real student or college data is included.

- 300 imaginary students; names are random combinations of common first names and surnames.
- Attendance, marks and assignment scores are generated from hidden "ability" and "engagement" values plus random noise.
- **Target definition:** `final_result = Fail` is a *historical non-pass* (class 1); `Pass` is class 0.
- `final_result` was **generated by an artificial rule**: the chance of `Fail` rises when attendance, internal marks, assignment score and previous marks are low, plus a random term. Because outcomes were generated from the input columns, models trained on this file will look better than they would on real students. **Do not read the scores as real-world accuracy.**
- Data-quality problems were inserted on purpose so the cleaning page has something to show: missing values, out-of-range and non-numeric entries, exact duplicates, a repeated ID, inconsistent labels, extra spaces, and 12 students with no outcome yet (these are scored by the model rather than used for training).

Regenerate it with `python data/generate_sample_data.py`.

## 6. How the prediction works

1. **Target** - `final_result` mapped to 0 (Pass) / 1 (Fail). Records without an outcome are not used for training or evaluation.
2. **Features** - attendance, internal marks, assignment score, previous-semester marks and any subject `_marks` columns.
3. **Leakage prevention** - the target, student ID, name, and group labels (department, year, section) are never model inputs. Group labels are excluded to avoid proxy bias. Imputation and scaling live inside the pipeline, so they are fitted on training data only.
4. **Split** - stratified 75% training / 25% held-out test, fixed `random_state`.
5. **Pipeline** - median imputation, standard scaling, then the classifier.
6. **Models** - Logistic Regression, a depth-limited Decision Tree, Random Forest (100 trees, depth 6), and K-Nearest Neighbors (5 neighbors). Logistic Regression and trees use balanced class weights; KNN uses unweighted standardized neighbors. All candidates use the same split and training-only preprocessing.
7. **Evaluation** - precision, recall and F1 for the non-pass class, accuracy, ROC AUC and the confusion matrix, all calculated on the held-out test set. The page also shows the accuracy of always predicting the most common outcome, because accuracy alone is misleading on imbalanced data.
8. **Cross-validation** - stratified 5-fold on the training portion, only when enough records exist.
9. **Refusal when unsuitable** - fewer than 40 labelled records, fewer than 8 examples of an outcome, or no `final_result` column: the app explains what is missing and shows indicator counts instead of a model. It never invents predictions.

### Risk categories

The estimated probability of a non-pass is converted with thresholds you can change in **Risk settings**:

| Category | Default rule |
|---|---|
| Low Risk | below 50% |
| Moderate Risk | 50% up to below 75% |
| High Risk | 75% or above |

Because the models weight the less common outcome more heavily, scores are a *relative ranking*, not calibrated chances of failing. A score near 50% means the model sees a student as about equally similar to past passes and fails. That is why the defaults start at 50%. The Model Evaluation page shows what actually happened to test students in each category, so you can judge whether the thresholds are useful.

Students with a known outcome are scored **out-of-fold** (by a model that never saw them); students without one are scored by the trained model.

### Suggestions

Rule-based and visible: attendance below 75%, internal marks below 40, assignment score below 50, a drop of more than 10 points since the previous semester, and low subject marks. All thresholds are adjustable. Suggestions are phrased as offers of help and never infer personal circumstances.

## 7. Syllabus concepts demonstrated

| Concept | Where |
|---|---|
| Data acquisition and validation | `modules/data_loader.py` |
| Data cleaning, missing values, type conversion | `modules/data_cleaning.py` |
| Descriptive statistics, distributions | Dashboard, `modules/exploratory_analysis.py` |
| Correlation vs causation | Exploratory Analysis |
| Outlier detection (IQR rule) | Exploratory Analysis |
| Supervised classification, pipelines, train/test split | `modules/prediction.py` |
| Class imbalance, cross-validation, data leakage | `modules/prediction.py` |
| Precision, recall, F1, confusion matrix | `modules/model_evaluation.py` |
| Interpretability (coefficients, importances, rules) | Risk Prediction, `modules/recommendations.py` |
| Data storage | `modules/database.py` (SQLite) |
| Ethics, privacy, bias, oversight | Data Ethics page |

## 8. Privacy

- Everything runs locally; usage statistics are disabled in `.streamlit/config.toml`.
- Anonymous IDs are **on by default**: names are hidden and IDs become `ANON-0001`-style labels in tables and downloads.
- The optional SQLite file `data/students.db` leaves out names unless you tick a box, and is excluded from git by `.gitignore`.
- Only use real student data if you are authorised to do so.

## 9. Suggested 5-minute demonstration

1. **Dashboard** - headline metrics; change a Department filter and watch every card and chart update.
2. **Data Cleaning** - show the duplicate, the out-of-range value and the `"absent"` entry in the change log.
3. **Exploratory Analysis** - Relationships tab: point out the correlation and the causation warning.
4. **Risk Prediction** - enter a low-attendance, low-marks student; show the category, contributing factors and suggestions; move a threshold in **Risk settings**.
5. **Model Evaluation** - explain precision vs recall, the confusion matrix, and the synthetic-data warning.
6. **Data Ethics** - human oversight and why the sample scores are not real-world accuracy.

## 10. Limitations

- Four baseline models are compared, with fixed hyper-parameters; no tuning or probability calibration is performed.
- Results come from a single train/test split and will vary with different data.
- Subject columns must end in `_marks` and use a 0-100 scale.
- Group error audits are available when sufficient held-out data exists; they do not prove fairness.
- Academic records omit context such as health, circumstances and support received. Correlation is not causation.
- Not intended for real students without institutional approval and human review.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the same Python environment you launch from |
| Page keeps showing old results after editing code | Stop the server (Ctrl+C) and run `streamlit run app.py` again |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Uploaded CSV rejected | Read the error message; it names the missing or empty column |


## 12. Advanced demonstration pages

The existing pages remain available. New navigation entries use the same source dataset, theme and filters.

1. **What-If Simulator:** select a student record from the filtered selection, then move attendance, internal marks, assignment and previous-semester sliders (subject sliders appear when available). Each move recalculates the scenario immediately. Both baseline and scenario use the same fitted model; the baseline can differ from out-of-fold scores in Student Records. Missing baseline values use training medians. Estimates are not guaranteed outcomes or evidence that changing an indicator causes an outcome.
2. **Explainable predictions:** Risk Prediction and the simulator show exact coefficient-times-standardized-value contributions for Logistic Regression. Other models show one-at-a-time score sensitivity to replacing an indicator with its training median. Sensitivities are not additive or causal. Global coefficients, tree impurity importance, or training-set permutation F1 importance are also available. Correlated features can distort interpretation.
3. **Semester Progress Tracker:** upload **long-format history** with a positive integer `semester` column, one row per `student_id, semester`. Reuse the same student ID across semesters. All existing required columns are still required on each row; blank numeric cells are allowed when a column has some numeric data. Example layout:

   ```csv
   student_id,semester,attendance_percentage,internal_marks,assignment_score,previous_semester_marks,final_result
   ```

   Add your observed records beneath that header. The built-in sample has no semester history, so this page asks for it instead of inventing trends. Select a student and indicators to view observed improvements/declines. Ambiguous semester labels, missing IDs, and conflicting student-semester records require correction. Changes compare consecutive available semesters in the current filters; gaps are not interpolated. Anonymous IDs stay stable across a student's semesters.
4. **Model Comparison Lab:** all four classifiers share the same held-out records; view precision, recall, F1, accuracy, ROC AUC, confusion matrices and training cross-validation. Repeated student IDs stay together in holdout, cross-validation and out-of-fold scoring, even without a semester column. Grouped evaluation requires at least eight distinct students associated with each outcome; it uses one feasible fold of a seeded four-fold stratified group split (approximately 25% test). Imputation and scaling are always learned inside each training fold. The sidebar model defaults to best **training cross-validation F1**, with Logistic Regression as fallback; the test set is never used to select a model. Metrics use full data and ignore descriptive sidebar filters. Features must have been recorded before the final result; chronology cannot be verified automatically.
5. **Report Generator:** download a multi-page PDF with selected filters, dataset counts/statistics, charts, source-data quality calculation, held-out metrics for all four models, confusion matrices and limitations. PDF generation stays local using existing Matplotlib dependencies. PDFs contain aggregates rather than individual student identities. Download processed CSV for the filtered selection; it follows the anonymous-ID setting. Empty selections and unavailable labels still produce an explanatory report.
6. **Data Quality Score:** assesses original loaded data before cleaning and ignores sidebar filters. Score = `100 * (1 - mean(missing-cell rate, invalid-cell rate, extra-duplicate-row rate, conflicting-record rate))`. Four checks have equal weights (25%); this is an explicitly chosen checklist, not an empirically validated quality or accuracy measure. Blank cells count as missing; invalid academic values and unknown nonblank outcome labels count as invalid. Conflict keys use student plus semester when available, otherwise student alone. Checks can overlap. The table gives issue counts, denominators and rates; Data Cleaning retains its detailed audit trail.
7. **Class and Subject Comparison:** choose an academic grouping and available subject/core indicators. Interactive heatmaps display actual means; accompanying counts show how many nonmissing records support each cell. Subject columns must end in `_marks`. All sidebar filters apply. Means weight semester records; comparisons are descriptive rather than causal or student rankings.
8. **Bias and Fairness Auditor:** evaluates full held-out predictions across available department, year or section groups, ignoring sidebar filters to avoid creating identifying slices. Sensitive attributes are not offered. Groups with fewer than **10 distinct held-out students** are suppressed, including names. False-positive / false-negative rates need 10 distinct students in the relevant actual-outcome denominator, otherwise they are unavailable. At least two eligible groups are needed for comparison. Missing group attributes are excluded. Counts and sample limitations accompany rates. Thresholds are a disclosure safeguard, not proof of statistical adequacy. Repeated semesters weight records; observed error gaps do not establish bias or fairness.

No additional Python dependencies are required. Run from this project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m unittest discover -s tests -t .
```

Advanced tests cover live simulator recalculation, model explanations, student-separated splits, known semester changes, quality arithmetic, group aggregates, fairness suppression, PDF generation and navigation for incomplete datasets.


## 13. Interface guide

The overview opens with a compact dataset-status strip, four headline metrics and an academic-support summary. Use the three action cards to open student records, try a what-if scenario, or export a report. The walkthrough explains the demonstration flow; chart tabs separate performance, subject trends and support indicators.

The sidebar keeps navigation visible and places controls in **Choose your data**, **Filter records**, and **Model & risk settings** panels. Source, anonymity and model availability remain visible in the status strip on every page. The simulator uses three numbered steps: choose a record, adjust indicators, compare estimates. Report downloads have separate PDF and CSV descriptions.

The navy / teal theme uses consistent chart cards, readable spacing, text labels alongside risk colours, keyboard focus indicators, and layouts that adapt to narrow screens. Existing navigation routes and analytical calculations are preserved.
