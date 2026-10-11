"""Presenting model evaluation results.

Concepts (Principles of Data Science): *evaluation metrics* (precision, recall,
F1), the *confusion matrix*, *held-out test data*, *model interpretability*
and *honest reporting of limitations*. All numbers come from the held-out test
split produced in ``prediction.train_models``; nothing is hard-coded.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed inside Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.config import RISK_HIGH, RISK_LOW, RISK_MODERATE, pretty
from modules.prediction import risk_category
from modules.theme import BORDER, CORAL, INK, INK_MUTED, SURFACE, TEAL, style_figure


def metrics_table(trained: dict) -> pd.DataFrame:
    """One row per model with test-set metrics for the non-pass class."""
    rows = []
    for name, result in trained["results"].items():
        m = result["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision (Fail)": m["precision"],
            "Recall (Fail)": m["recall"],
            "F1 (Fail)": m["f1"],
            "ROC AUC": m["roc_auc"],
        })
    return pd.DataFrame(rows)


def confusion_matrix_figure(matrix: np.ndarray, model_name: str):
    """Matplotlib confusion matrix with counts in every cell."""
    fig, ax = plt.subplots(figsize=(4.4, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(matrix, cmap="GnBu")
    labels = ["Pass", "Fail"]
    ax.set_xticks([0, 1], labels, color=INK_MUTED)
    ax.set_yticks([0, 1], labels, color=INK_MUTED)
    ax.set_xlabel("Predicted outcome", color=INK_MUTED)
    ax.set_ylabel("Actual outcome", color=INK_MUTED)
    ax.set_title(f"{model_name}: test-set confusion matrix", color=INK, fontsize=10)
    peak = matrix.max() if matrix.size else 1
    for (row, col), count in np.ndenumerate(matrix):
        # GnBu is light for small values and dark blue for large ones, so large counts
        # need light text and small counts need dark text to stay readable.
        dark_cell = peak > 0 and count / peak > 0.5
        ax.text(col, row, str(count), ha="center", va="center", fontsize=14, fontweight="bold",
                color="#ffffff" if dark_cell else "#0b1220")
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.tight_layout()
    return fig


def confusion_breakdown(matrix: np.ndarray) -> dict:
    """Name the four cells of the 2x2 confusion matrix."""
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "true_pass": tn, "false_alarm": fp, "missed_fail": fn, "caught_fail": tp,
    }


def importance_figure(table: pd.DataFrame, model_name: str) -> go.Figure:
    """Horizontal bar chart of coefficients or feature importances."""
    data = table.copy()
    data["Feature"] = data["Feature"].map(pretty)
    signed = "Coefficient" in data["Kind"].iloc[0]
    data = data.sort_values("Value" if signed else "Magnitude")
    colors = [CORAL if v > 0 else TEAL for v in data["Value"]] if signed else TEAL
    fig = go.Figure(go.Bar(
        x=data["Value"], y=data["Feature"], orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    if signed:
        fig.update_xaxes(title="Coefficient (standardised units)", zeroline=True)
        title = f"{model_name}: coefficients (red raises estimated risk, teal lowers it)"
    else:
        fig.update_xaxes(title="Relative importance")
        title = f"{model_name}: feature importance"
    fig = style_figure(fig, title, height=max(280, 110 + 40 * len(data)))
    # Let Plotly widen the left margin so long feature names are never clipped.
    fig.update_yaxes(title="", automargin=True)
    return fig


def threshold_table(trained: dict, model_name: str, moderate: float, high: float) -> pd.DataFrame:
    """What actually happened to test-set students in each risk category."""
    result = trained["results"][model_name]
    categories = [risk_category(p, moderate, high) for p in result["y_prob"]]
    frame = pd.DataFrame({"category": categories, "actual_fail": result["y_true"]})
    rows = []
    for category in (RISK_LOW, RISK_MODERATE, RISK_HIGH):
        subset = frame[frame["category"] == category]
        rows.append({
            "Risk category": category,
            "Test students": len(subset),
            "Actual non-pass": int(subset["actual_fail"].sum()),
            "Actual non-pass rate": float(subset["actual_fail"].mean()) if len(subset) else np.nan,
        })
    return pd.DataFrame(rows)


def evaluation_warnings(trained: dict, using_sample: bool) -> list[str]:
    """Plain-language warnings that should accompany any reported metric."""
    warnings = list(trained["notes"])
    if using_sample:
        warnings.append(
            "The sample dataset is synthetic: its outcomes were generated from an artificial "
            "rule plus random noise. These scores show that the workflow runs, NOT how well "
            "such a model would perform on real students."
        )
    baseline = trained["majority_baseline"]
    best_accuracy = max(r["metrics"]["accuracy"] for r in trained["results"].values())
    if best_accuracy <= baseline + 0.02:
        warnings.append(
            f"Accuracy ({best_accuracy:.0%}) is no better than always predicting the most "
            f"common outcome ({baseline:.0%}). The models add little over this baseline."
        )
    warnings.append(
        "Metrics describe one random train/test split of this dataset. They are estimates "
        "and will vary with different data or a different split."
    )
    warnings.append(
        "A model can only learn patterns present in past data. It cannot see personal "
        "circumstances, effort, or support received, and must not be the sole basis for "
        "any decision about a student."
    )
    return warnings
