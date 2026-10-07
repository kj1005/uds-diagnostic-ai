"""Evaluate retrieval, request validation, and test generation on a fixed dataset."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import RAGRetriever
from backend.test_generator import TestCaseGenerator
from backend.uds_validator import validate_request

EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "dataset.json"
JSON_REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"
MARKDOWN_REPORT_PATH = EVALUATION_DIR / "evaluation_report.md"


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def evaluate_retrieval(cases: list[dict[str, Any]], retriever: RAGRetriever) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            retrieved = retriever.retrieve(case["question"], top_k=3)
            sections = [chunk.section for chunk in retrieved]
            expected = normalize(case["expected_section"])
            hits = [normalize(section) == expected for section in sections]
            results.append({
                "id": case["id"], "question": case["question"],
                "expected_section": case["expected_section"],
                "retrieved_sections": sections,
                "hit_at_1": bool(hits and hits[0]),
                "hit_at_3": any(hits),
                "status": "passed" if any(hits) else "failed",
            })
        except Exception as exc:
            results.append({"id": case["id"], "question": case["question"],
                            "expected_section": case["expected_section"],
                            "retrieved_sections": [], "hit_at_1": False,
                            "hit_at_3": False, "status": "error", "error": str(exc)})
    count = len(results)
    return {"total": count,
            "hit_at_1": sum(result["hit_at_1"] for result in results) / count if count else 0.0,
            "hit_at_3": sum(result["hit_at_3"] for result in results) / count if count else 0.0,
            "hit_at_1_count": sum(result["hit_at_1"] for result in results),
            "hit_at_3_count": sum(result["hit_at_3"] for result in results),
            "cases": results}


def evaluate_validator(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            actual = validate_request(case["request"])
            checks = {
                "valid": actual.valid == case["expected_valid"],
                "service_id": actual.service_id == case["expected_service_id"],
                "nrc": actual.NRC == case["expected_nrc"],
            }
            results.append({"id": case["id"], "request": case["request"],
                            "expected": {"valid": case["expected_valid"],
                                         "service_id": case["expected_service_id"],
                                         "nrc": case["expected_nrc"]},
                            "actual": {"valid": actual.valid,
                                       "service_id": actual.service_id,
                                       "nrc": actual.NRC,
                                       "source_section": actual.source_section},
                            "checks": checks,
                            "status": "passed" if all(checks.values()) else "failed"})
        except Exception as exc:
            results.append({"id": case["id"], "request": case["request"],
                            "status": "error", "error": str(exc)})
    return {"total": len(results), "passed": sum(r["status"] == "passed" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "errors": sum(r["status"] == "error" for r in results), "cases": results}


def evaluate_generation(cases: list[dict[str, Any]], generator: TestCaseGenerator) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            generated = generator.generate(case["requirement"])
            actual_requests = [item.request for item in generated]
            actual_types = sorted({item.test_type.casefold() for item in generated})
            missing_requests = [request for request in case["expected_requests"]
                                if request not in actual_requests]
            missing_types = [kind for kind in case["required_types"]
                             if kind not in actual_types]
            passed = not missing_requests and not missing_types
            results.append({"id": case["id"], "requirement": case["requirement"],
                            "expected_requests": case["expected_requests"],
                            "generated_requests": actual_requests,
                            "expected_types": case["required_types"],
                            "generated_types": actual_types,
                            "missing_requests": missing_requests,
                            "missing_types": missing_types,
                            "source_sections": sorted({item.source_section for item in generated}),
                            "status": "passed" if passed else "failed"})
        except Exception as exc:
            results.append({"id": case["id"], "requirement": case["requirement"],
                            "status": "error", "error": str(exc)})
    return {"total": len(results), "passed": sum(r["status"] == "passed" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "errors": sum(r["status"] == "error" for r in results), "cases": results}


def markdown_report(report: dict[str, Any]) -> str:
    retrieval = report["retrieval"]
    lines = ["# UDS Diagnostic AI Evaluation Report", "",
             "Deterministic evaluation against the project’s synthetic knowledge base. "
             "No external LLM or ECU/CAN connection is used.", "",
             "## Summary", "",
             f"- Retrieval Hit@1: {retrieval['hit_at_1_count']}/{retrieval['total']} "
             f"({retrieval['hit_at_1']:.1%})",
             f"- Retrieval Hit@3: {retrieval['hit_at_3_count']}/{retrieval['total']} "
             f"({retrieval['hit_at_3']:.1%})",
             f"- Validator: {report['validator']['passed']}/{report['validator']['total']} passed",
             f"- Test generation: {report['test_generation']['passed']}/"
             f"{report['test_generation']['total']} passed", "",
             "## Retrieval", "", "| ID | Expected section | Hit@1 | Hit@3 | Retrieved sections |",
             "|---|---|---:|---:|---|"]
    for item in retrieval["cases"]:
        lines.append(f"| {item['id']} | {item['expected_section']} | "
                     f"{'Yes' if item['hit_at_1'] else 'No'} | "
                     f"{'Yes' if item['hit_at_3'] else 'No'} | "
                     f"{'; '.join(item['retrieved_sections']) or 'None'} |")
    lines += ["", "## Validator", "", "| ID | Request | Expected | Actual | Status |",
              "|---|---|---|---|---|"]
    for item in report["validator"]["cases"]:
        if "actual" in item:
            lines.append(f"| {item['id']} | `{item['request']}` | "
                         f"valid={item['expected']['valid']}, NRC={item['expected']['nrc']} | "
                         f"valid={item['actual']['valid']}, NRC={item['actual']['nrc']} | {item['status']} |")
        else:
            lines.append(f"| {item['id']} | `{item['request']}` | — | Error: {item.get('error', '')} | error |")
    lines += ["", "## Test generation", "", "| ID | Requirement | Generated requests | Status |",
              "|---|---|---|---|"]
    for item in report["test_generation"]["cases"]:
        lines.append(f"| {item['id']} | {item['requirement']} | "
                     f"{', '.join(f'`{request}`' for request in item.get('generated_requests', [])) or item.get('error', 'None')} | {item['status']} |")
    lines += ["", "## Method", "",
              "Retrieval compares each case’s expected section with the top three returned sections; "
              "Hit@1 and Hit@3 are reported as fractions over the 10 fixed retrieval questions. "
              "Validator cases check validity, service identifier, and NRC against fixed expected values. "
              "Generation cases check the required requests and positive/negative test types using the "
              "existing deterministic generator.", ""]
    return "\n".join(lines)


def run_evaluation(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    retriever = RAGRetriever()
    generator = TestCaseGenerator(retriever)
    return {
        "project": "AI-Powered UDS Diagnostic Assistant for Automated Test Case Generation",
        "dataset": dataset_path.name,
        "knowledge_base": "UDS Synthetic Knowledge Base",
        "deterministic": True,
        "external_llm_used": False,
        "retrieval": evaluate_retrieval(dataset["retrieval_cases"], retriever),
        "validator": evaluate_validator(dataset["validator_cases"]),
        "test_generation": evaluate_generation(dataset["generation_cases"], generator),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--json-out", type=Path, default=JSON_REPORT_PATH)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN_REPORT_PATH)
    args = parser.parse_args()
    try:
        report = run_evaluation(args.dataset)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        args.markdown_out.write_text(markdown_report(report), encoding="utf-8")
    except Exception as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"Evaluation complete. JSON: {args.json_out}; Markdown: {args.markdown_out}")
    print(f"Hit@1={report['retrieval']['hit_at_1']:.1%}, "
          f"Hit@3={report['retrieval']['hit_at_3']:.1%}; "
          f"validator={report['validator']['passed']}/{report['validator']['total']}; "
          f"generation={report['test_generation']['passed']}/{report['test_generation']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
