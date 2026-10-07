"""Validate UDS request formats using only the project's synthetic knowledge base."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

SYNTHETIC_KNOWLEDGE_BASE_NOTICE = (
    "Based on the project's synthetic demonstration knowledge base; "
    "not an official UDS specification. No ECU or CAN communication is performed."
)
INVALID_FORMAT_NRC = "0x13"
UNSUPPORTED_SUBFUNCTION_NRC = "0x12"
UNSUPPORTED_SERVICE_NRC = "0x11"
OUT_OF_RANGE_NRC = "0x31"
SECURITY_ACCESS_DENIED_NRC = "0x33"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    service_id: str | None
    service_name: str | None
    request: str
    validation_message: str
    expected_response: str | None
    negative_response: str | None
    NRC: str | None
    source_section: str
    knowledge_base_notice: str = SYNTHETIC_KNOWLEDGE_BASE_NOTICE


@dataclass(frozen=True)
class Service:
    name: str
    section: str
    subfunctions: dict[int, str] | None = None


SERVICES: dict[int, Service] = {
    0x10: Service(
        "Diagnostic Session Control",
        "SERVICE 0x10 — DIAGNOSTIC SESSION CONTROL",
        {0x01: "Default Session", 0x02: "Programming Session", 0x03: "Extended Diagnostic Session"},
    ),
    0x11: Service(
        "ECU Reset",
        "SERVICE 0x11 — ECU RESET",
        {0x01: "Hard Reset", 0x02: "Key Off On Reset", 0x03: "Soft Reset"},
    ),
    0x22: Service(
        "Read Data By Identifier",
        "SERVICE 0x22 — READ DATA BY IDENTIFIER",
    ),
    0x27: Service(
        "Security Access",
        "SERVICE 0x27 — SECURITY ACCESS",
        {0x01: "Request Seed", 0x02: "Send Key"},
    ),
    0x2E: Service(
        "Write Data By Identifier",
        "SERVICE 0x2E — WRITE DATA BY IDENTIFIER",
    ),
    0x31: Service(
        "Routine Control",
        "SERVICE 0x31 — ROUTINE CONTROL",
        {0x01: "Start Routine", 0x02: "Stop Routine", 0x03: "Request Routine Results"},
    ),
    0x3E: Service(
        "Tester Present",
        "SERVICE 0x3E — TESTER PRESENT",
        {0x00: "Zero subfunction"},
    ),
}

BYTE_TOKEN = re.compile(r"^(?:0[xX])?([0-9a-fA-F]{2})$")


def _format_request(request: str) -> tuple[list[int] | None, str]:
    tokens = request.split()
    if not tokens:
        return None, request.strip()

    parsed: list[int] = []
    for token in tokens:
        match = BYTE_TOKEN.fullmatch(token)
        if match is None:
            return None, request.strip()
        parsed.append(int(match.group(1), 16))
    return parsed, " ".join(f"{byte:02X}" for byte in parsed)


def _result(
    *,
    valid: bool,
    service_id: int | None,
    service: Service | None,
    request: str,
    message: str,
    expected_response: str | None = None,
    nrc: str | None = None,
    source_section: str = "REQUEST VALIDATION RULES",
) -> ValidationResult:
    negative_response = None
    if nrc is not None and service_id is not None:
        negative_response = f"7F {service_id:02X} {int(nrc, 16):02X}"
    return ValidationResult(
        valid=valid,
        service_id=f"0x{service_id:02X}" if service_id is not None else None,
        service_name=service.name if service is not None else None,
        request=request,
        validation_message=message,
        expected_response=expected_response,
        negative_response=negative_response,
        NRC=nrc,
        source_section=service.section if service is not None else source_section,
    )


def _failure(
    service_id: int | None,
    service: Service | None,
    request: str,
    message: str,
    nrc: str,
    source_section: str = "REQUEST VALIDATION RULES",
) -> ValidationResult:
    return _result(
        valid=False,
        service_id=service_id,
        service=service,
        request=request,
        message=message,
        nrc=nrc,
        source_section=source_section,
    )


def validate_request(
    request: str,
    *,
    security_access_granted: bool | None = None,
) -> ValidationResult:
    """Validate a whitespace-separated hexadecimal UDS request."""
    if not isinstance(request, str):
        raise TypeError("request must be a string containing hexadecimal bytes.")
    if security_access_granted is not None and not isinstance(
        security_access_granted, bool
    ):
        raise ValueError("security_access_granted must be True, False, or None.")

    values, normalized_request = _format_request(request)
    if values is None:
        first_token = request.strip().split(maxsplit=1)
        first_byte_match = BYTE_TOKEN.fullmatch(first_token[0]) if first_token else None
        service_id = (
            int(first_byte_match.group(1), 16) if first_byte_match is not None else None
        )
        service = SERVICES.get(service_id) if service_id is not None else None
        return _failure(
            service_id,
            service,
            normalized_request,
            "Request must contain whitespace-separated, two-digit hexadecimal bytes.",
            INVALID_FORMAT_NRC if service_id is not None else "0x13",
        )

    if not values:
        return _failure(
            None,
            None,
            normalized_request,
            "Request is empty and does not contain a Service Identifier.",
            INVALID_FORMAT_NRC,
        )

    service_id = values[0]
    service = SERVICES.get(service_id)
    if service is None:
        return _failure(
            service_id,
            None,
            normalized_request,
            "Service Identifier is not listed as supported in the synthetic knowledge base.",
            UNSUPPORTED_SERVICE_NRC,
            "UDS OVERVIEW",
        )

    if len(values) < 2:
        return _failure(
            service_id,
            service,
            normalized_request,
            "Request is incomplete; the documented request requires additional bytes.",
            INVALID_FORMAT_NRC,
        )

    if service_id in (0x10, 0x11, 0x27, 0x31, 0x3E):
        subfunction = values[1]
        assert service.subfunctions is not None
        if subfunction not in service.subfunctions:
            return _failure(
                service_id,
                service,
                normalized_request,
                f"Subfunction 0x{subfunction:02X} is not listed for {service.name}.",
                UNSUPPORTED_SUBFUNCTION_NRC,
            )

    if service_id == 0x10:
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message="Diagnostic session subfunction is listed in the knowledge base.",
            expected_response=f"50 {values[1]:02X} <session parameters>",
        )

    if service_id == 0x11:
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message="ECU reset subfunction is listed in the knowledge base.",
            expected_response=f"51 {values[1]:02X} <optional reset information>",
        )

    if service_id == 0x22:
        if (len(values) - 1) % 2 != 0:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Each Read Data By Identifier DID must contain two bytes; a DID is incomplete.",
                INVALID_FORMAT_NRC,
            )
        dids = [values[offset : offset + 2] for offset in range(1, len(values), 2)]
        if [0xFF, 0xFF] in dids:
            return _failure(
                service_id,
                service,
                normalized_request,
                "DID FFFF is the unsupported-DID example and is outside the supported range.",
                OUT_OF_RANGE_NRC,
            )
        if dids == [[0xF1, 0x90]]:
            expected_response = "62 F1 90 <VIN data>"
            message = "Request contains the documented two-byte demonstration DID F190."
        else:
            expected_response = "62 <DID> <data>"
            message = (
                "Request contains one or more complete two-byte DIDs. "
                "Support for these DID values is not established by the knowledge base."
            )
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message=message,
            expected_response=expected_response,
        )

    if service_id == 0x27:
        if values[1] == 0x02 and len(values) < 3:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Send Key request must include the key data shown in the knowledge-base example.",
                INVALID_FORMAT_NRC,
            )
        if values[1] == 0x02 and security_access_granted is False:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Security access condition is reported as not satisfied.",
                SECURITY_ACCESS_DENIED_NRC,
            )
        expected_response = (
            "67 01 <seed>" if values[1] == 0x01 else "67 02"
        )
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message=(
                "Security Access subfunction is listed. Key validity and delay "
                "conditions cannot be determined from the request bytes."
            ),
            expected_response=expected_response,
        )

    if service_id == 0x2E:
        if len(values) < 4:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Write Data By Identifier requires a two-byte DID and associated data.",
                INVALID_FORMAT_NRC,
            )
        if security_access_granted is False:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Security access condition is reported as not satisfied for this write request.",
                SECURITY_ACCESS_DENIED_NRC,
            )
        did = values[1:3]
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message=(
                f"Request contains a two-byte DID ({did[0]:02X}{did[1]:02X}) "
                "and associated data. DID support is not established by the "
                "knowledge base."
            ),
            expected_response=f"6E {did[0]:02X} {did[1]:02X}",
        )

    if service_id == 0x31:
        if len(values) < 4:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Routine Control requires a subfunction and a two-byte RID.",
                INVALID_FORMAT_NRC,
            )
        rid = values[2:4]
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message=(
                f"Request contains a listed Routine Control subfunction and "
                f"two-byte RID {rid[0]:02X}{rid[1]:02X}."
            ),
            expected_response=(
                f"71 {values[1]:02X} {rid[0]:02X} {rid[1]:02X} <optional data>"
            ),
        )

    if service_id == 0x3E:
        if len(values) != 2:
            return _failure(
                service_id,
                service,
                normalized_request,
                "Tester Present must use the supported format shown: SID and subfunction 0x00.",
                INVALID_FORMAT_NRC,
            )
        return _result(
            valid=True,
            service_id=service_id,
            service=service,
            request=normalized_request,
            message="Tester Present uses the documented zero subfunction.",
            expected_response="7E 00",
        )

    raise AssertionError(f"Validation is not implemented for supported service 0x{service_id:02X}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a hexadecimal UDS request against the synthetic project knowledge base."
    )
    parser.add_argument(
        "request",
        help='Hexadecimal bytes separated by spaces, for example "22 F1 90".',
    )
    parser.add_argument(
        "--security-access-denied",
        action="store_true",
        help=(
            "Report NRC 0x33 for a 0x27 Send Key or 0x2E write request when "
            "the caller knows the documented security condition is not satisfied."
        ),
    )
    args = parser.parse_args()
    try:
        result = validate_request(
            args.request,
            security_access_granted=False if args.security_access_denied else None,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
