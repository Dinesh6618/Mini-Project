"""Data Cleaning page: quality report and audit trail."""

from __future__ import annotations

import streamlit as st

from modules.data_loader import anonymize_frame
from views.common import csv_bytes, page_header, table


def render(ctx) -> None:
    report = ctx.report
    page_header(
        "Data Cleaning",
        "Nothing is changed silently. Every adjustment below was counted, explained and "
        "recorded in the change log.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows received", f"{report['rows_input']:,}")
    c2.metric("Duplicate rows removed", f"{report['duplicates_removed']:,}")
    c3.metric("Rows with repeated IDs", f"{report['duplicate_ids']:,}")
    c4.metric("Rows retained", f"{report['rows_output']:,}")

    if report["warnings"]:
        st.subheader("Warnings to review")
        for message in report["warnings"]:
            st.warning(message)
    else:
        st.success("No data-quality warnings were raised.")

    st.subheader("Missing and invalid values by column")
    if report["issues"].empty:
        st.success("No missing, non-numeric or out-of-range values were detected.")
    else:
        table(report["issues"])
        st.caption(
            "Invalid values (non-numeric, or outside the allowed range) are set to missing and "
            "listed in the change log. They are not guessed or corrected."
        )

    st.subheader("Operations performed")
    for operation in report["operations"]:
        st.markdown(f"- {operation}")

    st.subheader("Change log")
    log = report["change_log"]
    if log.empty:
        st.caption("No individual values were changed or removed.")
    else:
        shown = log.drop(columns="Student ID") if ctx.anonymize else log
        st.caption(
            f"{len(log)} change(s). 'Row' is the line number in the uploaded CSV (header = line 1)."
            + (" Student IDs are hidden because anonymous IDs are on." if ctx.anonymize else "")
        )
        table(shown)

    left, right = st.columns(2)
    with left:
        st.subheader("Column types after cleaning")
        table(report["dtypes"])
    with right:
        st.subheader("Prepared data preview")
        preview = anonymize_frame(ctx.cleaned) if ctx.anonymize else ctx.cleaned
        st.dataframe(preview.head(15), width="stretch", hide_index=True)
        st.caption("source_row is the line number of the record in the original CSV file.")

    export = anonymize_frame(ctx.cleaned) if ctx.anonymize else ctx.cleaned
    st.download_button(
        "Download prepared dataset (CSV)", csv_bytes(export),
        file_name="prepared_students.csv", mime="text/csv",
    )
    st.caption(
        "Missing values remain blank in this file. The prediction pipeline fills them with "
        "medians learned from the training split only, so no test information leaks in."
    )
