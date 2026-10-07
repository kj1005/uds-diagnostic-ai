"""Generate deterministic test-case templates grounded in retrieved UDS knowledge."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from backend.rag import RAGRetriever, RetrievedChunk
from backend.uds_validator import ValidationResult, validate_request

DEFAULT_TOP_K = 5
VALIDATION_EVIDENCE_QUERY = (
    "UDS request validation rules incomplete request invalid message format "
    "NRC 0x13 negative response structure"
)
SYNTHETIC_NOTICE = (
    "Generated only from retrieved project synthetic demonstration knowledge-base "
    "evidence; not an official UDS specification."
)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class GeneratedTestCase:
    test_case_id: str
    title: str
    requirement: str
    preconditions: list[str]
    request: str
    expected_response: str
    test_type: str
    pass_condition: str
    source_section: str
    source_filename: str
    rationale: str


@dataclass(frozen=True)
class CaseTemplate:
    title: str
    request: str
    test_type: str
    evidence_terms: tuple[str, ...]
    preconditions: tuple[str, ...]
    pass_condition: str
    rationale: str


@dataclass(frozen=True)
class ServicePlan:
    aliases: tuple[str, ...]
    positive: CaseTemplate
    negative: tuple[CaseTemplate, ...]


def _case(
    title: str,
    request: str,
    test_type: str,
    evidence_terms: tuple[str, ...],
    preconditions: tuple[str, ...],
    pass_condition: str,
    rationale: str,
) -> CaseTemplate:
    return CaseTemplate(
        title=title,
        request=request,
        test_type=test_type,
        evidence_terms=evidence_terms,
        preconditions=preconditions,
        pass_condition=pass_condition,
        rationale=rationale,
    )


SERVICE_PLANS: tuple[ServicePlan, ...] = (
    ServicePlan(
        aliases=("0x22", "read data by identifier", "read did", "vehicle identification number", "vin"),
        positive=_case(
            "Read supported VIN demonstration DID",
            "22 F1 90",
            "positive",
            ("22 F1 90", "62 F1 90"),
            ("DID F190 is the demonstration VIN DID described in the knowledge base.",),
            "The positive response begins with 62 F1 90.",
            "The retrieved knowledge base identifies F190 as its demonstration VIN DID and gives 22 F1 90 / 62 F1 90 as the request and positive response.",
        ),
        negative=(
            _case(
                "Read unsupported DID",
                "22 FF FF",
                "negative",
                ("22 FF FF", "7F 22 31"),
                ("DID FFFF is the unsupported-DID example in the synthetic knowledge base.",),
                "The response equals 7F 22 31.",
                "The retrieved sample scenario explicitly pairs request 22 FF FF with negative response 7F 22 31.",
            ),
            _case(
                "Read DID with incomplete request",
                "22 F1",
                "negative",
                (
                    "22 F1",
                    "NRC 0x13",
                    "incorrect message length",
                    "7F <Requested SID> <NRC>",
                ),
                ("The request contains only one byte of the two-byte DID.",),
                "The negative response indicates NRC 0x13 for incorrect message length or invalid format.",
                "The knowledge base labels 22 F1 as an incomplete DID request and associates it with NRC 0x13.",
            ),
        ),
    ),
    ServicePlan(
        aliases=("0x10", "diagnostic session control", "extended diagnostic session", "extended session"),
        positive=_case(
            "Enter Extended Diagnostic Session",
            "10 03",
            "positive",
            ("10 03", "50 03"),
            ("The synthetic knowledge base lists subfunction 0x03 as Extended Diagnostic Session.",),
            "The positive response begins with 50 03.",
            "The retrieved example requests Extended Diagnostic Session with 10 03 and expects a response beginning with 50 03.",
        ),
        negative=(),
    ),
    ServicePlan(
        aliases=("0x11", "ecu reset", "reset service"),
        positive=_case(
            "Request Hard Reset",
            "11 01",
            "positive",
            ("11 01", "51 <subfunction>"),
            ("The synthetic knowledge base lists subfunction 0x01 as Hard Reset.",),
            "The positive response follows 51 <subfunction> with optional reset information.",
            "The retrieved ECU Reset example gives request 11 01 for Hard Reset and documents a positive response in the 51 <subfunction> format.",
        ),
        negative=(),
    ),
    ServicePlan(
        aliases=("0x27", "security access", "request security seed"),
        positive=_case(
            "Request Security Access seed",
            "27 01",
            "positive",
            ("27 01", "67 01"),
            ("The synthetic knowledge base lists subfunction 0x01 as Request Seed.",),
            "The positive response begins with 67 01.",
            "The retrieved Security Access example gives request 27 01 and positive response 67 01 followed by seed data.",
        ),
        negative=(),
    ),
    ServicePlan(
        aliases=("0x3e", "tester present"),
        positive=_case(
            "Send Tester Present",
            "3E 00",
            "positive",
            ("3E 00", "7E 00"),
            ("The synthetic knowledge base lists subfunction 0x00 as the supported Zero subfunction.",),
            "The response equals 7E 00.",
            "The retrieved Tester Present example explicitly pairs request 3E 00 with positive response 7E 00.",
        ),
        negative=(),
    ),
)

INCOMPLETE_REQUEST_NEGATIVE = _case(
    "Reject incomplete request",
    "",
    "negative",
    (
        "Rule R009",
        "incomplete request should be rejected as an invalid message format",
        "0x13 - Incorrect Message Length Or Invalid Format",
        "7F <Requested SID> <NRC>",
    ),
    (),
    "The negative response uses NRC 0x13 for incorrect message length or invalid format.",
    "The retrieved validation rule requires incomplete requests to be rejected, and the retrieved NRC and response-structure sections define the corresponding response format.",
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _select_service(requirement: str) -> ServicePlan | None:
    normalized = requirement.casefold()
    matches = [
        (len(alias), plan)
        for plan in SERVICE_PLANS
        for alias in plan.aliases
        if alias in normalized
    ]
    if "did" in normalized and "read" in normalized:
        read_plan = next(plan for plan in SERVICE_PLANS if "read did" in plan.aliases)
        matches.append((len("read did"), read_plan))
    return max(matches, key=lambda match: match[0])[1] if matches else None


def _matching_evidence(
    template: CaseTemplate,
    evidence: list[RetrievedChunk],
) -> list[RetrievedChunk] | None:
    selected: list[RetrievedChunk] = []
    normalized_chunks = [(chunk, _normalize(chunk.chunk_text)) for chunk in evidence]
    for term in template.evidence_terms:
        normalized_term = _normalize(term)
        match = next(
            (
                chunk
                for chunk, chunk_text in normalized_chunks
                if normalized_term in chunk_text
            ),
            None,
        )
        if match is None:
            return None
        if match not in selected:
            selected.append(match)
    return selected


def _incomplete_negative_for(service_id: int) -> CaseTemplate:
    specific_rule = {
        0x10: "Rule R006",
        0x11: "Rule R007",
        0x27: "Rule R004",
        0x3E: "Rule R008",
    }.get(service_id)
    if specific_rule is None:
        specific_rule = "Rule R001"
    return CaseTemplate(
        title=f"Reject incomplete 0x{service_id:02X} request",
        request=f"{service_id:02X}",
        test_type="negative",
        evidence_terms=(specific_rule, *INCOMPLETE_REQUEST_NEGATIVE.evidence_terms),
        preconditions=(),
        pass_condition=INCOMPLETE_REQUEST_NEGATIVE.pass_condition,
        rationale=INCOMPLETE_REQUEST_NEGATIVE.rationale,
    )


def _combine_unique_chunks(
    primary: list[RetrievedChunk],
    additional: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    unique: dict[tuple[str, str, str], RetrievedChunk] = {}
    for chunk in (*primary, *additional):
        unique[(chunk.source_filename, chunk.section, chunk.chunk_text)] = chunk
    return list(unique.values())


def _make_case(
    template: CaseTemplate,
    requirement: str,
    evidence: list[RetrievedChunk],
    case_number: int,
) -> GeneratedTestCase | None:
    used_evidence = _matching_evidence(template, evidence)
    if used_evidence is None:
        return None

    validation: ValidationResult = validate_request(template.request)
    if template.test_type == "positive":
        if not validation.valid or validation.expected_response is None:
            return None
        expected_response = validation.expected_response
    else:
        if validation.valid or validation.negative_response is None:
            return None
        # Ensure the actual validator result is supported by the retrieved evidence.
        evidence_text = _normalize(" ".join(chunk.chunk_text for chunk in used_evidence))
        expected_nrc = validation.NRC
        if expected_nrc is None:
            return None
        exact_response = _normalize(validation.negative_response)
        response_format_grounded = (
            expected_nrc == "0x13"
            and "7f <requested sid> <nrc>" in evidence_text
            and "0x13 - incorrect message length or invalid format" in evidence_text
        )
        if exact_response not in evidence_text and not response_format_grounded:
            return None
        expected_response = validation.negative_response

    return GeneratedTestCase(
        test_case_id=f"TC_{case_number:03d}",
        title=template.title,
        requirement=requirement,
        preconditions=list(template.preconditions),
        request=template.request,
        expected_response=expected_response,
        test_type=template.test_type,
        pass_condition=template.pass_condition,
        source_section="; ".join(dict.fromkeys(chunk.section for chunk in used_evidence)),
        source_filename="; ".join(
            dict.fromkeys(chunk.source_filename for chunk in used_evidence)
        ),
        rationale=template.rationale,
    )


class TestCaseGenerator:
    """Generate grounded test cases using the local retriever and UDS validator."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = RAGRetriever()
        return self._retriever

    def generate(self, requirement: str, top_k: int = DEFAULT_TOP_K) -> list[GeneratedTestCase]:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("requirement must be a non-empty natural-language query.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        requirement = requirement.strip()
        service_plan = _select_service(requirement)
        if service_plan is None:
            raise ValueError(
                "Could not identify a supported UDS service in the requirement. "
                "Mention a service name or identifier such as VIN/DID, Diagnostic "
                "Session Control, ECU Reset, Security Access, or Tester Present."
            )

        evidence = self.retriever.retrieve(requirement, top_k=top_k)
        if not evidence:
            raise ValueError("The retriever returned no knowledge-base evidence.")

        all_templates = (service_plan.positive, *service_plan.negative)
        cases = [
            case
            for template in all_templates
            if (case := _make_case(template, requirement, evidence, 1)) is not None
        ]

        if not any(case.test_type == "negative" for case in cases):
            service_id_text = next(
                (alias for alias in service_plan.aliases if alias.startswith("0x")),
                None,
            )
            if service_id_text is not None:
                service_id = int(service_id_text, 16)
                additional = self.retriever.retrieve(
                    f"{service_id_text} incomplete request invalid message format "
                    f"Rule R009 NRC 0x13 {VALIDATION_EVIDENCE_QUERY}",
                    top_k=top_k,
                )
                evidence = _combine_unique_chunks(evidence, additional)

                if service_plan.negative:
                    for template in service_plan.negative:
                        case = _make_case(template, requirement, evidence, len(cases) + 1)
                        if case is not None and not any(
                            existing.request == case.request for existing in cases
                        ):
                            cases.append(case)
                else:
                    incomplete_template = _incomplete_negative_for(service_id)
                    case = _make_case(
                        incomplete_template,
                        requirement,
                        evidence,
                        len(cases) + 1,
                    )
                    if case is not None:
                        cases.append(case)

        if not any(case.test_type == "positive" for case in cases):
            raise ValueError(
                "Retrieved knowledge-base evidence did not support a concrete positive "
                "request and expected response for this service. No test cases were generated."
            )
        if not any(case.test_type == "negative" for case in cases):
            raise ValueError(
                "Retrieved knowledge-base evidence did not support a negative test for "
                "this service. No unsupported response has been invented."
            )

        return [
            GeneratedTestCase(
                test_case_id=f"TC_{index:03d}",
                title=case.title,
                requirement=case.requirement,
                preconditions=case.preconditions,
                request=case.request,
                expected_response=case.expected_response,
                test_type=case.test_type,
                pass_condition=case.pass_condition,
                source_section=case.source_section,
                source_filename=case.source_filename,
                rationale=case.rationale,
            )
            for index, case in enumerate(cases, start=1)
        ]


def generate_test_cases(
    requirement: str,
    top_k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> list[GeneratedTestCase]:
    """Convenience API for generating evidence-grounded test cases."""
    return TestCaseGenerator(retriever).generate(requirement, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic UDS test cases from retrieved local knowledge."
    )
    parser.add_argument("requirement", help="Natural-language test requirement.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of semantic matches to retrieve (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    try:
        cases = generate_test_cases(args.requirement, top_k=args.top_k)
    except (ValueError, RuntimeError) as exc:
        parser.exit(status=1, message=f"Test generation failed: {exc}\n")
    print(
        json.dumps(
            {
                "requirement": args.requirement,
                "knowledge_base_notice": SYNTHETIC_NOTICE,
                "test_cases": [asdict(case) for case in cases],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
