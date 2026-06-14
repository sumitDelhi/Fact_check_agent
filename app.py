"""FactCheck AI Streamlit application."""

from __future__ import annotations

import os
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from services.claim_extractor import ClaimExtractor
from services.pdf_parser import PDFParser
from services.report_generator import ReportGenerator
from services.verifier import Verifier
from utils.constants import (
    APP_NAME,
    DEFAULT_MODEL,
    ERROR_API_KEY,
    STATUS_COLORS,
    STATUS_UNVERIFIABLE,
    SUPPORTED_MODELS,
)
from utils.helpers import calculate_claim_statistics, truncate_text, validate_pdf_file

load_dotenv()
BASE_DIR = Path(__file__).parent


def configure_cloud_secrets() -> None:
    """Expose Streamlit Cloud secrets through os.environ for service classes."""
    for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        try:
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            continue


configure_cloud_secrets()

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(BASE_DIR / "assets" / "logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1180px;}
    .app-title {font-size: 2.2rem; font-weight: 760; margin: 0;}
    .muted {color: #64748b; margin-top: .2rem;}
    .status-card {
        border: 1px solid #e2e8f0;
        border-left-width: 6px;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: .75rem;
        background: #ffffff;
    }
    .status-card:hover {box-shadow: 0 4px 18px rgba(2,6,23,0.08)}
    .app-header {display:flex; align-items:center; gap:12px}
    .app-badge {background: linear-gradient(90deg,#06b6d4,#7c3aed); color: #fff; padding:6px 10px; border-radius:999px; font-weight:600}
    .muted {color: #64748b; margin-top: .2rem;}
    .source-link {font-size: .92rem;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "extracted_text": "",
        "extracted_claims": [],
        "verification_results": [],
        "document_name": "",
        "document_metadata": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def has_required_keys() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY"))


def sidebar_controls():
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF document",
            type=["pdf"],
            help="Maximum size: 25 MB.",
        )

        st.header("Verification Settings")
        model = st.selectbox(
            "Model",
            SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in SUPPORTED_MODELS else 0,
        )
        remove_duplicates = st.toggle("Remove duplicate claims", value=True)
        max_claims = st.slider("Maximum claims", 5, 100, 40, step=5)
        min_confidence = st.slider("Minimum confidence to show", 0, 100, 0, step=5)

        st.caption("Uses Tavily live web search and OpenAI reasoning. Store API keys in `.env` locally or Streamlit secrets in production.")

    return uploaded_file, model, remove_duplicates, max_claims, min_confidence


def reset_for_new_file(uploaded_file) -> None:
    if uploaded_file and uploaded_file.name != st.session_state.document_name:
        st.session_state.extracted_text = ""
        st.session_state.extracted_claims = []
        st.session_state.verification_results = []
        st.session_state.document_metadata = {}
        st.session_state.document_name = uploaded_file.name


def extract_text_and_claims(uploaded_file, model: str, remove_duplicates: bool, max_claims: int) -> None:
    is_valid, error = validate_pdf_file(uploaded_file)
    if not is_valid:
        st.error(error)
        return

    parser = PDFParser()
    progress = st.progress(0, text="Reading PDF...")
    text, pdf_error = parser.extract_text(uploaded_file)
    if pdf_error:
        progress.empty()
        st.error(pdf_error)
        return

    st.session_state.extracted_text = text
    st.session_state.document_metadata = parser.get_pdf_metadata(uploaded_file)
    progress.progress(35, text="Extracting factual claims...")

    extractor = ClaimExtractor(model=model)
    claims, claim_error = extractor.extract_and_process_claims(
        text,
        remove_duplicates=remove_duplicates,
        max_claims=max_claims,
    )
    progress.progress(100, text="Claims extracted.")
    progress.empty()

    if claim_error:
        st.error(claim_error)
        return

    st.session_state.extracted_claims = claims
    st.session_state.verification_results = []
    st.success(f"Extracted {len(claims)} factual claims.")


def verify_claims(model: str, min_confidence: int, selected_indices: list[int] | None = None) -> None:
    all_claims = st.session_state.extracted_claims
    if not all_claims:
        st.warning("Extract claims before starting verification.")
        return

    claims = all_claims
    if selected_indices is not None:
        claims = [all_claims[i] for i in selected_indices]

    verifier = Verifier(model=model)
    progress = st.progress(0, text="Starting verification...")
    status = st.empty()

    def on_progress(current: int, total: int) -> None:
        percent = int((current / total) * 100) if total else 0
        progress.progress(percent, text=f"Verified {current} of {total} claims")
        status.caption(f"Checking claim {min(current + 1, total)} of {total}")

    results, error = verifier.verify_claims(
        claims,
        progress_callback=on_progress,
        min_confidence=min_confidence,
    )
    progress.progress(100, text="Verification complete.")
    status.empty()

    if error:
        st.error(error)
        return

    # merge results into session state; if we verified a subset, replace only those
    if selected_indices is None:
        st.session_state.verification_results = results
    else:
        # keep original results and overwrite for selected indices
        existing = st.session_state.verification_results or [{} for _ in all_claims]
        for idx, res in zip(selected_indices, results):
            if idx < len(existing):
                existing[idx] = res
        st.session_state.verification_results = existing

    st.success(f"Verified {len(results)} claims.")


def render_upload_section(uploaded_file, model: str, remove_duplicates: bool, max_claims: int) -> None:
    st.subheader("PDF Upload")
    if not uploaded_file:
        st.info("Upload a PDF from the sidebar to begin.")
        return

    is_valid, error = validate_pdf_file(uploaded_file)
    if error:
        st.error(error)
    else:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"Selected: **{uploaded_file.name}** ({size_mb:.2f} MB)")

    if st.button("Extract Text and Claims", type="primary", use_container_width=True, disabled=not is_valid):
        extract_text_and_claims(uploaded_file, model, remove_duplicates, max_claims)

    metadata = st.session_state.document_metadata
    if metadata:
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", metadata.get("pages", 0))
        col2.metric("Size", f"{metadata.get('file_size_mb', 0):.2f} MB")
        col3.metric("Extracted characters", f"{len(st.session_state.extracted_text):,}")

    # Preview extracted text for easier inspection
    if st.session_state.extracted_text:
        with st.expander("Preview extracted text", expanded=False):
            st.text_area("", value=st.session_state.extracted_text, height=300)


def render_claims_section(model: str, min_confidence: int) -> None:
    st.subheader("Extracted Claims Table")
    claims = st.session_state.extracted_claims
    if not claims:
        st.info("Claims will appear here after extraction.")
        return

    # Filter and select claims to verify
    search = st.text_input("Filter claims", placeholder="Search claim text or type...")
    indexed = list(enumerate(claims))
    if search:
        indexed = [(i, c) for i, c in indexed if search.lower() in c.get("claim", "").lower() or search.lower() in str(c.get("type", "")).lower()]

    options = [f"{i+1}: {truncate_text(c['claim'], 100)}" for i, c in indexed]
    selected = st.multiselect("Select claims to verify (multiselect)", options, default=options[:5])
    # map selections back to indices
    selected_indices = [indexed[options.index(s)][0] for s in selected] if selected else []

    # Show a compact table of the filtered claims
    rows = [{"#": i + 1, "Claim": c["claim"], "Type": c.get("type", ""), "Status": "Pending"} for i, c in indexed]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 1])
    if col1.button("Verify Selected", type="primary", use_container_width=True, disabled=not selected_indices):
        verify_claims(model, min_confidence, selected_indices=selected_indices)
    if col2.button("Verify All (slow)", use_container_width=True):
        verify_claims(model, min_confidence)


def render_metrics(results: list[dict]) -> None:
    stats = calculate_claim_statistics(results)
    cols = st.columns(5)
    cols[0].metric("Claims", stats["total"])
    cols[1].metric("Verified", stats["verified"])
    cols[2].metric("Inaccurate", stats["inaccurate"])
    cols[3].metric("False", stats["false"])
    cols[4].metric("Avg Confidence", f"{stats['avg_confidence']}%")


def render_report_section() -> None:
    st.subheader("Verification Report")
    results = st.session_state.verification_results
    if not results:
        st.info("Verification results will appear here.")
        return

    render_metrics(results)
    st.divider()

    for index, result in enumerate(results, start=1):
        status = result.get("status", STATUS_UNVERIFIABLE)
        color = STATUS_COLORS.get(status, STATUS_COLORS[STATUS_UNVERIFIABLE])
        with st.expander(
            f"{index}. {status} | {result.get('confidence', 0)}% | {truncate_text(result.get('claim', ''), 100)}",
            expanded=index <= 3,
        ):
            st.markdown(
                f"<div class='status-card' style='border-left-color:{color}'>"
                f"<b>Claim:</b> {escape(str(result.get('claim', '')))}<br>"
                f"<b>Type:</b> {escape(str(result.get('type', '')))}<br>"
                f"<b>Status:</b> {escape(str(status))}<br>"
                f"<b>Confidence:</b> {result.get('confidence', 0)}%<br>"
                f"<b>Reason:</b> {escape(str(result.get('explanation', '')))}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if result.get("key_finding"):
                st.write(f"**Key finding:** {result['key_finding']}")
            if result.get("sources"):
                st.write("**Sources**")
                for source in result["sources"]:
                    st.markdown(f"- [{source.get('title', 'Source')}]({source.get('url', '')})")
            if result.get("evidence_snippet"):
                with st.popover("Evidence snippet"):
                    st.text(result["evidence_snippet"])


def render_download_section() -> None:
    st.subheader("Download Report")
    results = st.session_state.verification_results
    if not results:
        st.info("Run verification before downloading reports.")
        return

    generator = ReportGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2, col3 = st.columns(3)

    csv_data, csv_error = generator.generate_csv_report(results)
    col1.download_button(
        "Download CSV",
        data=csv_data if not csv_error else "",
        file_name=f"factcheck_report_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=bool(csv_error),
    )

    pdf_data, pdf_error = generator.generate_pdf_report(results)
    col2.download_button(
        "Download PDF",
        data=pdf_data if not pdf_error else b"",
        file_name=f"factcheck_report_{timestamp}.pdf",
        mime="application/pdf",
        use_container_width=True,
        disabled=bool(pdf_error),
    )

    md_data, md_error = generator.generate_markdown_report(results)
    col3.download_button(
        "Download Markdown",
        data=md_data if not md_error else "",
        file_name=f"factcheck_report_{timestamp}.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=bool(md_error),
    )


def main() -> None:
    init_state()

    logo_path = BASE_DIR / "assets" / "logo.png"
    title_col, copy_col = st.columns([0.12, 0.88])
    with title_col:
        if logo_path.exists():
            st.image(str(logo_path), width=72)
    with copy_col:
        st.markdown("<p class='app-title'>FactCheck AI</p>", unsafe_allow_html=True)
        st.markdown(
            "<p class='muted'>Automated PDF claim verification for reports, whitepapers, articles, and company documents.</p>",
            unsafe_allow_html=True,
        )

    if not has_required_keys():
        st.error(ERROR_API_KEY)
        st.stop()

    uploaded_file, model, remove_duplicates, max_claims, min_confidence = sidebar_controls()
    reset_for_new_file(uploaded_file)

    tab_upload, tab_claims, tab_report, tab_download = st.tabs([
        "PDF Upload",
        "Extracted Claims",
        "Verification Report",
        "Download Report",
    ])

    with tab_upload:
        render_upload_section(uploaded_file, model, remove_duplicates, max_claims)

    with tab_claims:
        render_claims_section(model, min_confidence)

    with tab_report:
        render_report_section()

    with tab_download:
        render_download_section()


if __name__ == "__main__":
    main()
