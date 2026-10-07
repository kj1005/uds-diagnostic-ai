import unittest

from backend.uds_validator import validate_request


class ValidateRequestTests(unittest.TestCase):
    def test_read_supported_did(self) -> None:
        result = validate_request("22 F1 90")
        self.assertTrue(result.valid)
        self.assertEqual(result.service_id, "0x22")
        self.assertEqual(result.expected_response, "62 F1 90 <VIN data>")
        self.assertIsNone(result.negative_response)

    def test_read_unsupported_did(self) -> None:
        result = validate_request("22 FF FF")
        self.assertFalse(result.valid)
        self.assertEqual(result.negative_response, "7F 22 31")
        self.assertEqual(result.NRC, "0x31")

    def test_read_incomplete_did(self) -> None:
        result = validate_request("22 F1")
        self.assertFalse(result.valid)
        self.assertEqual(result.negative_response, "7F 22 13")
        self.assertEqual(result.NRC, "0x13")

    def test_enter_extended_diagnostic_session(self) -> None:
        result = validate_request("10 03")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "50 03 <session parameters>")

    def test_ecu_reset(self) -> None:
        result = validate_request("11 01")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "51 01 <optional reset information>")

    def test_tester_present(self) -> None:
        result = validate_request("3E 00")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "7E 00")

    def test_security_access_seed_request(self) -> None:
        result = validate_request("27 01")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "67 01 <seed>")

    def test_security_access_key_request(self) -> None:
        result = validate_request("27 02 12 34")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "67 02")

    def test_write_data_by_identifier(self) -> None:
        result = validate_request("2E F1 90 01")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "6E F1 90")

    def test_routine_control(self) -> None:
        result = validate_request("31 01 12 34")
        self.assertTrue(result.valid)
        self.assertEqual(result.expected_response, "71 01 12 34 <optional data>")

    def test_unsupported_service(self) -> None:
        result = validate_request("99")
        self.assertFalse(result.valid)
        self.assertEqual(result.negative_response, "7F 99 11")
        self.assertEqual(result.NRC, "0x11")
        self.assertEqual(result.source_section, "UDS OVERVIEW")

    def test_unsupported_session_subfunction(self) -> None:
        result = validate_request("10 04")
        self.assertFalse(result.valid)
        self.assertEqual(result.negative_response, "7F 10 12")

    def test_explicitly_unsatisfied_security_condition(self) -> None:
        result = validate_request("2E F1 90 01", security_access_granted=False)
        self.assertFalse(result.valid)
        self.assertEqual(result.negative_response, "7F 2E 33")
        self.assertEqual(result.NRC, "0x33")

    def test_non_hexadecimal_byte_is_invalid_format(self) -> None:
        result = validate_request("22 FG")
        self.assertFalse(result.valid)
        self.assertEqual(result.negative_response, "7F 22 13")
        self.assertEqual(result.NRC, "0x13")


if __name__ == "__main__":
    unittest.main()
