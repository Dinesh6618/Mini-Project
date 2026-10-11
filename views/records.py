"""Student Records page: searchable table, downloads and optional SQLite save."""

from __future__ import annotations

import streamlit as st

from modules import database
from modules.config import ID_COLUMN, NAME_COLUMN
from modules.data_loader import anonymize_frame
from modules.recommendations import academic_recommendations
from views.common import csv_bytes, empty_view, page_header


def _display_columns(view) -> list[str]:
    preferred = [
        "student_id", "student_name", "department", "year", "section",
        "attendance_percentage", "internal_marks", "assignment_score",
        "previous_semester_marks",
    ]
    columns = [c for c in preferred if c in view.columns]
    columns += [c for c in view.columns if c not in columns and c not in {"source_row"}]
    return columns


def render(ctx) -> None:
    page_header(
        "Student Records",
        f"{ctx.source_label}. Search the table, use the sidebar filters, and download the "
        "processed data.",
    )
    if ctx.anonymize:
        st.info("Anonymous IDs are on: student names are hidden and IDs are replaced with ANON-#### labels.")
    elif NAME_COLUMN in ctx.cleaned.columns:
        st.warning(
            "Anonymous IDs are off, so student names are visible. Only enable this when you are "
            "authorised to view identifiable records."
        )
    if empty_view(ctx):
        return

    view = ctx.view
    query = st.text_input("Search records", placeholder="Type any part of an ID, department, result...")
    shown = view
    if query.strip():
        text = view.astype("string").fillna("")
        mask = text.apply(lambda col: col.str.contains(query.strip(), case=False, regex=False)).any(axis=1)
        shown = view[mask]
    st.caption(f"Showing {len(shown):,} of {len(ctx.scored):,} records.")

    config = {}
    if "risk_probability" in shown.columns:
        config["risk_probability"] = st.column_config.ProgressColumn(
            "Estimated non-pass probability", format="percent", min_value=0.0, max_value=1.0,
        )
    st.dataframe(
        shown[_display_columns(shown)], width="stretch", height=460, hide_index=True,
        column_config=config,
    )

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download displayed records (CSV)", csv_bytes(shown),
        file_name="student_records_displayed.csv", mime="text/csv",
        help="Respects the current filters, search and anonymity setting.",
    )
    full = ctx.scored.copy()
    if ctx.anonymize:
        full = anonymize_frame(full)
    c2.download_button(
        "Download full processed dataset (CSV)", csv_bytes(full),
        file_name="processed_students.csv", mime="text/csv",
        help="All cleaned records with risk estimates, ignoring filters.",
    )

    st.divider()
    st.subheader("Support suggestions for one student")
    st.caption("Suggestions come from simple thresholds on the indicators, not from the model.")
    options = shown[ID_COLUMN].astype(str).tolist() if ID_COLUMN in shown.columns else []
    if options:
        chosen = st.selectbox("Student", options, key="records_student")
        row = shown[shown[ID_COLUMN].astype(str) == chosen].iloc[0]
        tips = academic_recommendations(row.to_dict(), ctx.indicators)
        if tips:
            for tip in tips:
                st.write(f"- {tip}")
        else:
            st.write(
                "No indicator is below its threshold. Continue to use individual context and "
                "professional judgement."
            )

    st.divider()
    with st.expander("Save to local SQLite database (optional)"):
        st.caption(
            "Stores the prepared dataset in data/students.db on this computer so it can be "
            "reloaded from the sidebar. Nothing is sent anywhere. Student IDs are stored as "
            "they appear in the file (the anonymous labels are not saved); names are left out "
            "unless you tick the box."
        )
        include_names = st.checkbox("Include student names", value=False, key="db_names")
        left, right = st.columns(2)
        if left.button("Save prepared dataset"):
            try:
                saved = database.save_dataset(ctx.cleaned, include_names=include_names)
                st.success(f"Saved {saved:,} records to {database.DB_PATH.name}.")
            except Exception as exc:  # sqlite3.Error or OSError
                st.error(f"Could not save the database: {exc}")
        if database.database_exists() and right.button("Delete saved database"):
            database.delete_database()
            st.success("The saved database was deleted.")
