"""Download the current analysis and processed selection."""
import streamlit as st
from modules.reporting import analysis_pdf
from views.common import page_header


def render(ctx):
    page_header("Report Generator", "Download a local PDF summary and the processed records in the current filters.")
    st.write(f"Selected records: {len(ctx.view)} of {len(ctx.cleaned)} processed records.")
    with st.expander("Review selected filters"):
        st.write(ctx.selected_filters)
    st.caption("PDF includes charts, statistics, source-data quality, all model metrics and confusion matrices when available, filters and limitations. Evaluation metrics describe the full held-out split. No student-level identifiers are included in the PDF.")
    st.subheader("Analysis summary")
    st.caption("A PDF for your presentation: charts, statistics, model evaluation and limitations.")
    st.download_button("Download analysis PDF", data=analysis_pdf(ctx), file_name="academic_analysis.pdf", mime="application/pdf")
    st.divider()
    st.subheader("Processed student records")
    st.caption("A CSV for further analysis. Includes only records in the current filters.")
    if not ctx.anonymize:
        st.warning("Anonymous IDs are off: CSV can include student identifiers and names. Enable anonymous IDs in the sidebar to hide these fields.")
    st.download_button("Download processed CSV", data=ctx.view.to_csv(index=False).encode("utf-8-sig"), file_name="processed_students.csv", mime="text/csv")
