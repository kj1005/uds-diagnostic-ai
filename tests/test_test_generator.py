import unittest

from backend.rag import RetrievedChunk
from backend.test_generator import (
    SYNTHETIC_NOTICE,
    TestCaseGenerator,
    generate_test_cases,
)


def chunk(section: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_text=text,
        section=section,
        distance=0.1,
        similarity=0.9,
        source_filename="UDS_Synthetic_Knowledge_Base.txt",
        chunk_id=f"test-{section.lower().replace(' ', '-')}",
    )


class FakeRetriever:
    def __init__(self, evidence: list[RetrievedChunk]) -> None:
        self.evidence = evidence
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.queries.append((query, top_k))
        return self.evidence


SCENARIOS = chunk(
    "SAMPLE DIAGNOSTIC SCENARIOS",
    """
    Request: 22 F1 90
    Expected: Positive response beginning with 62 F1 90.
    Request: 22 FF FF
    Expected: Negative response: 7F 22 31
    Request: 22 F1
    Expected: Negative response condition associated with incorrect message length, NRC 0x13.
    Request: 10 03
    Expected: Positive response beginning with 50 03.
    Request: 3E 00
    Expected: Positive response: 7E 00
    """,
)
SESSION = chunk(
    "SERVICE 0x10 — DIAGNOSTIC SESSION CONTROL",
    """
    Diagnostic Session Control
    Example subfunctions:
    0x01 - Default Session
    0x02 - Programming Session
    0x03 - Extended Diagnostic Session
    Example request: 10 03
    Positive response: 50 <subfunction> <session parameters>
    Example: 50 03 ...
    Possible negative response conditions include unsupported subfunction and incorrect message length.
    """,
)
TESTER_PRESENT = chunk(
    "SERVICE 0x3E — TESTER PRESENT",
    """
    Tester Present
    Example request: 3E 00
    Positive response: 7E 00
    Subfunction: 0x00 - Zero subfunction
    Possible negative response: 7F 3E <NRC>
    """,
)
VALIDATION_RULES = chunk(
    "REQUEST VALIDATION RULES",
    """
    Rule R004: Service 0x27 requires a valid security access subfunction.
    Rule R006: Service 0x10 requires a valid diagnostic-session subfunction.
    Rule R007: Service 0x11 requires a valid ECU-reset subfunction.
    Rule R008: Service 0x3E requires the Tester Present subfunction in the supported format.
    Rule R009: An incomplete request should be rejected as an invalid message format.
    """,
)
NEGATIVE_RESPONSE_CODES = chunk(
    "NEGATIVE RESPONSE CODES",
    """
    A negative response has the general structure:
    7F <Requested SID> <NRC>
    0x13 - Incorrect Message Length Or Invalid Format
    """,
)
ECU_RESET = chunk(
    "SERVICE 0x11 — ECU RESET",
    """
    ECU Reset
    Example request: 11 01
    Positive response: 51 <subfunction> <optional reset information>
    """,
)
SECURITY_ACCESS = chunk(
    "SERVICE 0x27 — SECURITY ACCESS",
    """
    Security Access
    Example seed request: 27 01
    Example positive response: 67 01 <seed>
    """,
)


class TestCaseGeneratorTests(unittest.TestCase):
    def test_read_vin_generates_grounded_positive_and_negative_cases(self) -> None:
        retriever = FakeRetriever([SCENARIOS, NEGATIVE_RESPONSE_CODES])
        cases = generate_test_cases(
            "Generate tests for reading the VIN DID",
            retriever=retriever,
        )

        self.assertEqual(
            [case.request for case in cases],
            ["22 F1 90", "22 FF FF", "22 F1"],
        )
        self.assertEqual([case.test_type for case in cases], ["positive", "negative", "negative"])
        self.assertEqual(cases[0].expected_response, "62 F1 90 <VIN data>")
        self.assertEqual(cases[1].expected_response, "7F 22 31")
        self.assertEqual(cases[2].expected_response, "7F 22 13")
        self.assertEqual(cases[0].source_section, "SAMPLE DIAGNOSTIC SCENARIOS")
        self.assertIn("UDS_Synthetic_Knowledge_Base.txt", cases[0].source_filename)
        self.assertEqual(len({case.test_case_id for case in cases}), len(cases))
        self.assertGreater(len(retriever.queries), 0)

    def test_extended_session_generates_positive_and_incomplete_negative(self) -> None:
        retriever = FakeRetriever([SESSION, VALIDATION_RULES, NEGATIVE_RESPONSE_CODES])
        cases = TestCaseGenerator(retriever).generate(
            "Generate tests to enter Extended Diagnostic Session"
        )

        self.assertEqual([case.request for case in cases], ["10 03", "10"])
        self.assertEqual([case.test_type for case in cases], ["positive", "negative"])
        self.assertEqual(cases[0].expected_response, "50 03 <session parameters>")
        self.assertEqual(cases[1].expected_response, "7F 10 13")
        self.assertIn("REQUEST VALIDATION RULES", cases[1].source_section)

    def test_tester_present_generates_positive_and_incomplete_negative(self) -> None:
        retriever = FakeRetriever(
            [TESTER_PRESENT, VALIDATION_RULES, NEGATIVE_RESPONSE_CODES]
        )
        cases = TestCaseGenerator(retriever).generate("Generate tests for Tester Present")

        self.assertEqual([case.request for case in cases], ["3E 00", "3E"])
        self.assertEqual([case.test_type for case in cases], ["positive", "negative"])
        self.assertEqual(cases[0].expected_response, "7E 00")
        self.assertEqual(cases[1].expected_response, "7F 3E 13")

    def test_reset_and_security_access_examples_are_supported(self) -> None:
        evidence = [
            ECU_RESET,
            SECURITY_ACCESS,
            VALIDATION_RULES,
            NEGATIVE_RESPONSE_CODES,
        ]
        reset_cases = generate_test_cases("Generate tests for ECU Reset", retriever=FakeRetriever(evidence))
        security_cases = generate_test_cases(
            "Generate tests for Security Access", retriever=FakeRetriever(evidence)
        )

        self.assertEqual([case.request for case in reset_cases], ["11 01", "11"])
        self.assertEqual(
            reset_cases[0].expected_response,
            "51 01 <optional reset information>",
        )
        self.assertEqual([case.request for case in security_cases], ["27 01", "27"])
        self.assertEqual(security_cases[0].expected_response, "67 01 <seed>")

    def test_does_not_generate_case_without_retrieved_evidence(self) -> None:
        retriever = FakeRetriever([])
        with self.assertRaisesRegex(ValueError, "no knowledge-base evidence"):
            TestCaseGenerator(retriever).generate("Generate tests for Tester Present")

    def test_does_not_invent_an_unsupported_test_for_unknown_service(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not identify a supported UDS service"):
            TestCaseGenerator(FakeRetriever([SCENARIOS])).generate(
                "Generate tests for an undocumented service"
            )

    def test_case_has_expected_schema_and_synthetic_notice(self) -> None:
        case = generate_test_cases(
            "Read DID",
            retriever=FakeRetriever([SCENARIOS, NEGATIVE_RESPONSE_CODES]),
        )[0]

        self.assertEqual(
            set(case.__dataclass_fields__),
            {
                "test_case_id",
                "title",
                "requirement",
                "preconditions",
                "request",
                "expected_response",
                "test_type",
                "pass_condition",
                "source_section",
                "source_filename",
                "rationale",
            },
        )
        self.assertIn("synthetic demonstration", SYNTHETIC_NOTICE)


if __name__ == "__main__":
    unittest.main()
