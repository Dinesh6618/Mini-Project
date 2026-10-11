"""Exploratory data analysis: statistics, tables and Plotly figures.

Concepts (Principles of Data Science): *descriptive statistics*,
*distributions*, *correlation*, *outlier detection (IQR rule)* and
*visual communication*. Functions return figures or tables; the Streamlit
views decide where to place them. A function returns ``None`` when the columns
it needs are not in the dataset, so views can show an informative empty state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.config import ID_COLUMN, TARGET_COLUMN, pretty, subject_columns
from modules.theme import (
    AMBER, CORAL, DIVERGING, INK_MUTED, OUTCOME_COLORS, TEAL, style_figure,
)

PERFORMANCE_BANDS = [
    (0, 40, "Below 40"),
    (40, 60, "40 - 59"),
    (60, 75, "60 - 74"),
    (75, 101, "75 and above"),
]


def outcome_labels(frame: pd.DataFrame) -> pd.Series | None:
    """Pass / Fail / Unknown per student, or None if there is no outcome column."""
    if TARGET_COLUMN not in frame.columns:
        return None
    return frame[TARGET_COLUMN].fillna("Unknown").astype(str)


def _has(frame: pd.DataFrame, *columns: str) -> bool:
    return all(c in frame.columns for c in columns) and len(frame) > 0


# ----------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------
def histogram(frame: pd.DataFrame, column: str, by_outcome: bool = True, nbins: int = 20) -> go.Figure | None:
    """Distribution of one numeric column, optionally split by historical outcome."""
    if not _has(frame, column) or frame[column].dropna().empty:
        return None
    data = frame.dropna(subset=[column]).copy()
    labels = outcome_labels(data) if by_outcome else None
    if labels is not None:
        data["Historical result"] = labels
        fig = px.histogram(
            data, x=column, color="Historical result", nbins=nbins, barmode="stack",
            color_discrete_map=OUTCOME_COLORS,
            category_orders={"Historical result": ["Pass", "Fail", "Unknown"]},
            labels={column: pretty(column)},
        )
    else:
        fig = px.histogram(data, x=column, nbins=nbins, labels={column: pretty(column)},
                           color_discrete_sequence=[TEAL])
    fig.update_yaxes(title="Number of students")
    return style_figure(fig, f"Distribution of {pretty(column).lower()}")


def performance_band_counts(frame: pd.DataFrame, column: str = "internal_marks") -> pd.DataFrame | None:
    """Number of students in each internal-marks band."""
    if not _has(frame, column):
        return None
    values = frame[column].dropna()
    rows = [
        {"Band": label, "Students": int(((values >= low) & (values < high)).sum())}
        for low, high, label in PERFORMANCE_BANDS
    ]
    return pd.DataFrame(rows)


def performance_band_chart(frame: pd.DataFrame, column: str = "internal_marks") -> go.Figure | None:
    table = performance_band_counts(frame, column)
    if table is None or table["Students"].sum() == 0:
        return None
    fig = px.bar(table, x="Band", y="Students", text="Students", color_discrete_sequence=[TEAL])
    fig.update_traces(textposition="outside", textfont_color=INK_MUTED, cliponaxis=False)
    fig.update_xaxes(title=f"{pretty(column)} band (out of 100)")
    fig.update_yaxes(title="Number of students")
    return style_figure(fig, "Overall performance distribution")


# ----------------------------------------------------------------------
# Relationships
# ----------------------------------------------------------------------
def scatter_with_trend(
    frame: pd.DataFrame, x: str, y: str, title: str, hover: list[str] | None = None,
) -> tuple[go.Figure, float | None, int] | None:
    """Scatter plot of two columns with a least-squares line.

    Returns ``(figure, pearson_r, n_pairs)`` or ``None`` when columns are absent.
    """
    if not _has(frame, x, y):
        return None
    data = frame.dropna(subset=[x, y]).copy()
    if len(data) < 2:
        return None
    labels = outcome_labels(data)
    hover_columns = [c for c in (hover or []) if c in data.columns]
    if labels is not None:
        data["Historical result"] = labels
        fig = px.scatter(
            data, x=x, y=y, color="Historical result", color_discrete_map=OUTCOME_COLORS,
            hover_data=hover_columns, opacity=0.8,
            labels={x: pretty(x), y: pretty(y)},
        )
    else:
        fig = px.scatter(
            data, x=x, y=y, hover_data=hover_columns, opacity=0.8,
            labels={x: pretty(x), y: pretty(y)}, color_discrete_sequence=[TEAL],
        )
    r = None
    if data[x].nunique() > 1 and data[y].nunique() > 1:
        r = float(np.corrcoef(data[x], data[y])[0, 1])
        slope, intercept = np.polyfit(data[x], data[y], 1)
        xs = np.array([data[x].min(), data[x].max()])
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines", name="Trend line",
            line=dict(color=INK_MUTED, width=2, dash="dash"),
        ))
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#111c2d")), selector=dict(mode="markers"))
    return style_figure(fig, title), r, len(data)


def previous_vs_current(frame: pd.DataFrame) -> tuple[go.Figure, float | None, int] | None:
    """Previous-semester marks against current internal marks with y = x guide."""
    result = scatter_with_trend(
        frame, "previous_semester_marks", "internal_marks",
        "Previous semester marks vs current internal marks",
        hover=[ID_COLUMN],
    )
    if result is None:
        return None
    fig, r, n = result
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="No change (y = x)",
        line=dict(color=CORAL, width=1.5, dash="dot"),
    ))
    return fig, r, n


def change_in_marks(frame: pd.DataFrame) -> pd.Series | None:
    if not _has(frame, "internal_marks", "previous_semester_marks"):
        return None
    return (frame["internal_marks"] - frame["previous_semester_marks"]).dropna()


def change_histogram(frame: pd.DataFrame) -> go.Figure | None:
    change = change_in_marks(frame)
    if change is None or change.empty:
        return None
    fig = px.histogram(change.rename("Change"), x="Change", nbins=20, color_discrete_sequence=[TEAL])
    fig.add_vline(x=0, line_color=INK_MUTED, line_dash="dash")
    fig.update_xaxes(title="Current internal marks minus previous semester marks")
    fig.update_yaxes(title="Number of students")
    return style_figure(fig, "Change in marks since previous semester")


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Pearson correlations among academic measures (identifiers excluded)."""
    columns = [c for c in academic_columns(frame) if frame[c].nunique() > 1]
    if len(columns) < 2:
        return None
    return frame[columns].corr(numeric_only=True)


# Numeric columns that are bookkeeping or model outputs, not academic measures.
NON_ACADEMIC_NUMERIC = {"year", "source_row", "risk_probability"}


def academic_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns that measure academic performance or attendance."""
    return [
        c for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and c not in NON_ACADEMIC_NUMERIC
    ]


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    labels = [pretty(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(title="Pearson r", tickfont=dict(color=INK_MUTED)),
        hovertemplate="%{y} vs %{x}: r = %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, "Correlation between academic measures", height=460)


def describe_correlation(r: float | None) -> str:
    """Plain-language strength of a correlation coefficient."""
    if r is None or np.isnan(r):
        return "not available"
    size = abs(r)
    strength = (
        "very weak" if size < 0.2 else "weak" if size < 0.4 else
        "moderate" if size < 0.6 else "strong" if size < 0.8 else "very strong"
    )
    return f"{strength} {'positive' if r > 0 else 'negative'}"


def attendance_correlations(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Pearson r between attendance and each academic score."""
    if not _has(frame, "attendance_percentage"):
        return None
    targets = [c for c in ["internal_marks", "assignment_score", "previous_semester_marks"]
               + subject_columns(frame) if c in frame.columns]
    rows = []
    for column in targets:
        pair = frame[["attendance_percentage", column]].dropna()
        if len(pair) < 3 or pair[column].nunique() < 2 or pair["attendance_percentage"].nunique() < 2:
            continue
        r = float(pair["attendance_percentage"].corr(pair[column]))
        rows.append({
            "Measure": pretty(column), "Pearson r": round(r, 3),
            "Strength": describe_correlation(r), "Students": len(pair),
        })
    return pd.DataFrame(rows) if rows else None


# ----------------------------------------------------------------------
# Assignments and subjects
# ----------------------------------------------------------------------
def assignment_by_outcome(frame: pd.DataFrame) -> go.Figure | None:
    if not _has(frame, "assignment_score"):
        return None
    data = frame.dropna(subset=["assignment_score"]).copy()
    labels = outcome_labels(data)
    if labels is None or labels.nunique() < 1:
        return None
    data["Historical result"] = labels
    fig = px.box(
        data, x="Historical result", y="assignment_score", color="Historical result",
        color_discrete_map=OUTCOME_COLORS, points="outliers",
        labels={"assignment_score": pretty("assignment_score")},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Historical result")
    return style_figure(fig, "Assignment scores by historical result")


def subject_summary(frame: pd.DataFrame) -> pd.DataFrame | None:
    columns = subject_columns(frame)
    if not columns or len(frame) == 0:
        return None
    rows = []
    for column in columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        rows.append({
            "Subject": pretty(column).removesuffix(" marks"), "Average": values.mean(),
            "Median": values.median(), "Lowest": values.min(), "Highest": values.max(),
            "Below 40": int((values < 40).sum()), "Students": len(values),
        })
    return pd.DataFrame(rows) if rows else None


def subject_bar_chart(frame: pd.DataFrame) -> go.Figure | None:
    table = subject_summary(frame)
    if table is None:
        return None
    fig = px.bar(
        table, x="Subject", y="Average", text=table["Average"].round(1),
        color_discrete_sequence=[TEAL],
    )
    fig.update_traces(textposition="outside", textfont_color=INK_MUTED, cliponaxis=False)
    fig.update_yaxes(title="Average marks (out of 100)", range=[0, 100])
    return style_figure(fig, "Subject-wise average performance")


def subject_box_chart(frame: pd.DataFrame) -> go.Figure | None:
    columns = subject_columns(frame)
    if not columns:
        return None
    long = frame[columns].rename(columns={c: pretty(c).removesuffix(" marks") for c in columns})
    long = long.melt(var_name="Subject", value_name="Marks").dropna()
    if long.empty:
        return None
    fig = px.box(long, x="Subject", y="Marks", points="outliers", color_discrete_sequence=[TEAL])
    fig.update_yaxes(title="Marks (out of 100)")
    return style_figure(fig, "Spread of marks by subject")


def performance_trend_by_group(frame: pd.DataFrame, group: str) -> go.Figure | None:
    """Average previous-semester vs current internal marks for each group."""
    if not _has(frame, group, "previous_semester_marks", "internal_marks"):
        return None
    summary = (
        frame.groupby(group, observed=True)[["previous_semester_marks", "internal_marks"]]
        .mean().dropna(how="all").reset_index()
    )
    if summary.empty:
        return None
    summary[group] = summary[group].astype(str)
    long = summary.melt(id_vars=group, var_name="Measure", value_name="Average marks")
    long["Measure"] = long["Measure"].map(pretty)
    fig = px.bar(
        long, x=group, y="Average marks", color="Measure", barmode="group",
        color_discrete_sequence=[AMBER, TEAL],
        labels={group: group.replace("_", " ").title()},
    )
    fig.update_yaxes(range=[0, 100], title="Average marks (out of 100)")
    return style_figure(fig, f"Performance trend by {group.replace('_', ' ')}")


# ----------------------------------------------------------------------
# Outliers
# ----------------------------------------------------------------------
def find_outliers(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Values outside 1.5 x IQR of each measure (a prompt for human review)."""
    if columns is None:
        columns = [c for c in ["attendance_percentage", "internal_marks", "assignment_score",
                               "previous_semester_marks"] if c in frame.columns]
        columns += subject_columns(frame)
    rows = []
    for column in columns:
        values = frame[column].dropna()
        if len(values) < 8:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        flagged = frame.index[(frame[column] < low) | (frame[column] > high)]
        for index in flagged:
            rows.append({
                "Student ID": frame.at[index, ID_COLUMN] if ID_COLUMN in frame.columns else index,
                "Measure": pretty(column), "Value": float(frame.at[index, column]),
                "Typical range (IQR rule)": f"{max(low, 0):.1f} - {min(high, 100):.1f}",
                "Direction": "Unusually low" if frame.at[index, column] < low else "Unusually high",
            })
    return pd.DataFrame(
        rows, columns=["Student ID", "Measure", "Value", "Typical range (IQR rule)", "Direction"]
    )


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame | None:
    columns = academic_columns(frame)
    if not columns:
        return None
    stats = frame[columns].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats.index = [pretty(c) for c in stats.index]
    return stats.rename(columns={"50%": "median"}).round(2)
