"""Local PDF reports using Matplotlib; no network or extra PDF dependency."""
from io import BytesIO
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from modules.config import CORE_FEATURES
from modules.advanced import quality_assessment
from modules.model_evaluation import metrics_table, evaluation_warnings


def analysis_pdf(ctx):
    output = BytesIO()
    with PdfPages(output) as pdf:
        def text_pages(title, lines):
            wrapped = []
            for line in lines:
                wrapped.extend(textwrap.wrap(str(line), width=105) or [""])
            for start in range(0, max(1, len(wrapped)), 43):
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(.08, .95, title, fontsize=16, va="top")
                fig.text(.08, .90, "\n".join(wrapped[start:start+43]), fontsize=10, va="top", linespacing=1.5)
                pdf.savefig(fig)
                plt.close(fig)
        score, quality = quality_assessment(ctx.raw)
        lines = [ctx.source_label, f"Loaded records: {len(ctx.raw)}; processed: {len(ctx.cleaned)}; selected: {len(ctx.view)}.",
                 f"Selected model: {ctx.model_name or 'Unavailable'}; moderate threshold: {ctx.moderate:.0%}; high threshold: {ctx.high:.0%}.",
                 "Selected filters: " + str(ctx.selected_filters),
                 "Anonymous identifiers: " + str(ctx.anonymize),
                 f"Source-data quality: {score:.1f}/100 = 100 * (1 - mean of four issue rates). Equal weights are a disclosed policy choice."]
        lines += [f"{r['Check']}: {r['Issues']} / {r['Denominator']} ({r['Issue rate']:.1%})" for _, r in quality.iterrows()]
        lines += ["Missing / invalid cells, duplicates and conflicting student-semester rows can overlap. Missing optional columns are not penalized.",
                  "Statistics below use selected records; model metrics always use the full held-out split."]
        text_pages("Student academic analysis summary", lines)
        columns = [c for c in CORE_FEATURES if c in ctx.view]
        if len(ctx.view):
            stats = ctx.view[columns].describe().round(2)
            text_pages("Selected dataset statistics", ["Indicator order: " + ", ".join(columns)] + [f"{idx}: " + ", ".join(f"{c}={row[c]}" for c in columns) for idx, row in stats.iterrows()])
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ctx.view.internal_marks.dropna().plot.hist(ax=axes[0], bins=15, color="#1fa595")
            axes[0].set_title("Selected internal marks")
            axes[0].set_xlabel("Marks / 100")
            axes[1].scatter(ctx.view.attendance_percentage, ctx.view.internal_marks, alpha=.5, color="#1fa595")
            axes[1].set_xlabel("Attendance (%)")
            axes[1].set_ylabel("Internal marks")
            axes[1].set_title("Selected attendance vs marks")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        if ctx.has_model:
            trained = ctx.trained
            summary = metrics_table(trained)
            text_pages("Model comparison: held-out evaluation", [f"Train records: {trained['n_train']}; test records: {trained['n_test']}; student-separated split: {trained['grouped_split']}.",
                       "All models share the split. Preprocessing is fitted on training data only. Precision, recall and F1 refer to historical non-pass (Fail).",
                       "Logistic Regression / trees use balanced class weights; KNN uses distance in standardized academic features."] +
                       [str(row.to_dict()) for _, row in summary.round(4).iterrows()])
            fig, axes = plt.subplots(2, 2, figsize=(9, 8))
            for ax, (name, result) in zip(axes.flat, trained["results"].items()):
                matrix = result["confusion_matrix"]
                ax.imshow(matrix, cmap="GnBu")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
                ax.set_xticks([0, 1], ["Pass", "Fail"])
                ax.set_yticks([0, 1], ["Pass", "Fail"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        limitations = ["Estimates are not guaranteed outcomes, calibrated probabilities, or evidence of causation. Human oversight and student context are required.",
                       "No personal student records appear in this PDF. CSV downloads follow the anonymous-ID setting and current filters.",
                       "Historical indicators must have been available before final outcomes. The application cannot verify source chronology.",
                       "Quality scoring describes data issues, not predictive validity. Group comparisons require adequate independent samples.",
                       "Sidebar filters affect descriptive statistics and charts only; training and evaluation use the full dataset."]
        limitations += evaluation_warnings(ctx.trained, ctx.using_sample) if ctx.has_model else [ctx.training_error or "No model available."]
        limitations += ctx.report["warnings"]
        text_pages("Limitations and interpretation", limitations)
    return output.getvalue()
