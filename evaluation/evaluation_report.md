# UDS Diagnostic AI Evaluation Report

Deterministic evaluation against the project’s synthetic knowledge base. No external LLM or ECU/CAN connection is used.

## Summary

- Retrieval Hit@1: 9/10 (90.0%)
- Retrieval Hit@3: 10/10 (100.0%)
- Validator: 8/8 passed
- Test generation: 3/3 passed

## Retrieval

| ID | Expected section | Hit@1 | Hit@3 | Retrieved sections |
|---|---|---:|---:|---|
| R01 | UDS OVERVIEW | Yes | Yes | UDS OVERVIEW; SERVICE 0x2E — WRITE DATA BY IDENTIFIER; SERVICE 0x31 — ROUTINE CONTROL |
| R02 | SERVICE 0x10 — DIAGNOSTIC SESSION CONTROL | Yes | Yes | SERVICE 0x10 — DIAGNOSTIC SESSION CONTROL; SAMPLE DIAGNOSTIC SCENARIOS; SAMPLE TEST CASE TEMPLATES |
| R03 | SERVICE 0x11 — ECU RESET | Yes | Yes | SERVICE 0x11 — ECU RESET; SAMPLE TEST CASE TEMPLATES; NEGATIVE RESPONSE CODES |
| R04 | SERVICE 0x22 — READ DATA BY IDENTIFIER | No | Yes | SYNTHETIC DATASET LIMITATIONS; SERVICE 0x22 — READ DATA BY IDENTIFIER; SAMPLE TEST CASE TEMPLATES |
| R05 | SERVICE 0x27 — SECURITY ACCESS | Yes | Yes | SERVICE 0x27 — SECURITY ACCESS; SERVICE 0x2E — WRITE DATA BY IDENTIFIER; NEGATIVE RESPONSE CODES |
| R06 | SERVICE 0x2E — WRITE DATA BY IDENTIFIER | Yes | Yes | SERVICE 0x2E — WRITE DATA BY IDENTIFIER; SERVICE 0x22 — READ DATA BY IDENTIFIER; POSITIVE AND NEGATIVE RESPONSE STRUCTURE |
| R07 | SERVICE 0x31 — ROUTINE CONTROL | Yes | Yes | SERVICE 0x31 — ROUTINE CONTROL; SERVICE 0x10 — DIAGNOSTIC SESSION CONTROL; SERVICE 0x3E — TESTER PRESENT |
| R08 | SERVICE 0x3E — TESTER PRESENT | Yes | Yes | SERVICE 0x3E — TESTER PRESENT; SAMPLE TEST CASE TEMPLATES; SERVICE 0x10 — DIAGNOSTIC SESSION CONTROL |
| R09 | NEGATIVE RESPONSE CODES | Yes | Yes | NEGATIVE RESPONSE CODES; POSITIVE AND NEGATIVE RESPONSE STRUCTURE; SERVICE 0x31 — ROUTINE CONTROL |
| R10 | POSITIVE AND NEGATIVE RESPONSE STRUCTURE | Yes | Yes | POSITIVE AND NEGATIVE RESPONSE STRUCTURE; NEGATIVE RESPONSE CODES; SERVICE 0x31 — ROUTINE CONTROL |

## Validator

| ID | Request | Expected | Actual | Status |
|---|---|---|---|---|
| V01 | `22 F1 90` | valid=True, NRC=None | valid=True, NRC=None | passed |
| V02 | `22 FF FF` | valid=False, NRC=0x31 | valid=False, NRC=0x31 | passed |
| V03 | `10 03` | valid=True, NRC=None | valid=True, NRC=None | passed |
| V04 | `10 7F` | valid=False, NRC=0x12 | valid=False, NRC=0x12 | passed |
| V05 | `22 F1` | valid=False, NRC=0x13 | valid=False, NRC=0x13 | passed |
| V06 | `3E 00` | valid=True, NRC=None | valid=True, NRC=None | passed |
| V07 | `99 00` | valid=False, NRC=0x11 | valid=False, NRC=0x11 | passed |
| V08 | `27 02` | valid=False, NRC=0x13 | valid=False, NRC=0x13 | passed |

## Test generation

| ID | Requirement | Generated requests | Status |
|---|---|---|---|
| G01 | Generate tests for reading the VIN DID | `22 F1 90`, `22 FF FF` | passed |
| G02 | Generate tests for Diagnostic Session Control | `10 03`, `10` | passed |
| G03 | Generate tests for Tester Present | `3E 00`, `3E` | passed |

## Method

Retrieval compares each case’s expected section with the top three returned sections; Hit@1 and Hit@3 are reported as fractions over the 10 fixed retrieval questions. Validator cases check validity, service identifier, and NRC against fixed expected values. Generation cases check the required requests and positive/negative test types using the existing deterministic generator.
