"""Streamlit demonstration UI for the synthetic UDS Diagnostic AI project."""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit may launch this file with frontend/ as the import root. Add the
# project root so the existing backend package remains importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from backend.rag import RAGRetriever
from backend.test_generator import TestCaseGenerator
from backend.uds_validator import validate_request


DISCLAIMER = (
    "This is an academic demonstration using a synthetic UDS knowledge base. "
    "It does not communicate with a real ECU or CAN bus. All outputs require "
    "engineer review."
)


st.set_page_config(
    page_title="AI-Powered UDS Diagnostic Assistant",
    page_icon="🔧",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the local knowledge base…")
def get_retriever() -> RAGRetriever:
    """Create one reusable local retriever when a retrieval action is requested."""
    return RAGRetriever()


@st.cache_resource(show_spinner="Preparing the deterministic test generator…")
def get_test_generator() -> TestCaseGenerator:
    """Reuse the same retriever for grounded test generation."""
    return TestCaseGenerator(get_retriever())


st.title("AI-Powered UDS Diagnostic Assistant")
st.write(
    "Explore synthetic UDS knowledge, validate hexadecimal diagnostic requests, "
    "and generate evidence-grounded test case templates."
)
st.warning(DISCLAIMER, icon="⚠️")

with st.sidebar:
    st.header("Project setup")
    st.markdown("**Embedding model**  \n`all-MiniLM-L6-v2`")
    st.markdown("**Vector store**  \nChromaDB")
    st.markdown("**Execution mode**  \nLocal / No ECU communication")
    st.divider()
    st.subheader("Architecture / flow")
    st.caption(
        "User Query → Embedding → ChromaDB Retrieval → Evidence → "
        "Deterministic UDS Validation → Grounded Test Generation"
    )
    st.caption("Synthetic demonstration knowledge base only. No ECU/CAN bus connection.")


ask_tab, validate_tab, generate_tab = st.tabs(
    ["Ask UDS", "Validate Request", "Generate Test Cases"]
)

with ask_tab:
    st.subheader("Ask the knowledge base")
    with st.form("ask_uds_form"):
        question = st.text_input(
            "Diagnostic question",
            placeholder="What is the positive response for reading VIN DID F190?",
        )
        top_k = st.slider("Number of evidence chunks (top-k)", 1, 10, 5)
        ask_submitted = st.form_submit_button("Retrieve evidence", type="primary")

    if ask_submitted:
        if not question.strip():
            st.info("Enter a diagnostic question to retrieve evidence.")
        else:
            try:
                evidence = get_retriever().retrieve(question.strip(), top_k=top_k)
                if not evidence:
                    st.info("No matching evidence was returned from the synthetic knowledge base.")
                else:
                    st.success(
                        "Knowledge-base-grounded evidence. The app does not use an external LLM."
                    )
                    for index, item in enumerate(evidence, start=1):
                        with st.expander(f"Evidence {index}: {item.section}", expanded=index == 1):
                            st.markdown(item.chunk_text)
                            first, second = st.columns(2)
                            first.metric("Similarity", f"{item.similarity:.4f}")
                            second.metric("Distance", f"{item.distance:.4f}")
                            st.markdown(f"**Source section:** {item.section}")
                            st.caption(f"Source filename: {item.source_filename}")
            except Exception as exc:
                st.error(f"Evidence retrieval failed: {exc}")

with validate_tab:
    st.subheader("Validate a hexadecimal UDS request")
    with st.form("validate_request_form"):
        request = st.text_input(
            "UDS request bytes",
            value="22 F1 90",
            help="Enter whitespace-separated hexadecimal bytes, for example 22 F1 90.",
        )
        validate_submitted = st.form_submit_button("Validate request", type="primary")

    if validate_submitted:
        try:
            result = validate_request(request)
            if result.valid:
                st.success("Valid according to the synthetic knowledge base")
            else:
                st.error("Invalid according to the synthetic knowledge base")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Service**")
                st.write(
                    f"{result.service_name or 'Unknown'} "
                    f"({result.service_id or 'N/A'})"
                )
                st.markdown("**Request**")
                st.code(result.request or "(empty)")
                st.markdown("**Expected response**")
                st.code(result.expected_response or "Not established")
            with col2:
                st.markdown("**Negative response**")
                st.code(result.negative_response or "None")
                st.markdown("**NRC**")
                st.code(result.NRC or "None")
                st.markdown("**Source section**")
                st.write(result.source_section)
            st.markdown("**Validation message**")
            st.info(result.validation_message)
            st.caption(result.knowledge_base_notice)
        except (TypeError, ValueError) as exc:
            st.error(f"Request validation failed: {exc}")
        except Exception as exc:
            st.error(f"The validator could not process this request: {exc}")

with generate_tab:
    st.subheader("Generate grounded test case templates")
    with st.form("generate_test_cases_form"):
        requirement = st.text_input(
            "Requirement",
            placeholder="Generate tests for reading the VIN DID",
        )
        generation_top_k = st.slider("Evidence chunks (top-k)", 1, 10, 5)
        generate_submitted = st.form_submit_button("Generate test cases", type="primary")

    if generate_submitted:
        if not requirement.strip():
            st.info("Enter a requirement to generate test cases.")
        else:
            try:
                cases = get_test_generator().generate(
                    requirement.strip(), top_k=generation_top_k
                )
                st.success(f"Generated {len(cases)} test cases from retrieved evidence.")
                for case in cases:
                    kind = case.test_type.casefold()
                    label = "Positive" if kind == "positive" else "Negative"
                    with st.expander(f"{case.test_case_id} · {label} · {case.title}", expanded=True):
                        st.markdown(f"**Type:** {label}")
                        st.markdown(f"**Request:** `{case.request}`")
                        st.markdown(f"**Expected response:** `{case.expected_response}`")
                        st.markdown(f"**Source section:** {case.source_section}")
                        st.markdown(f"**Rationale:** {case.rationale}")
                        if case.preconditions:
                            st.markdown("**Preconditions**")
                            for precondition in case.preconditions:
                                st.markdown(f"- {precondition}")
                        st.markdown(f"**Pass condition:** {case.pass_condition}")
                        st.caption(f"Source filename: {case.source_filename}")
            except (ValueError, RuntimeError) as exc:
                st.error(f"Test case generation failed: {exc}")
            except Exception as exc:
                st.error(f"The local retrieval service could not complete generation: {exc}")

st.divider()
st.caption(DISCLAIMER)
