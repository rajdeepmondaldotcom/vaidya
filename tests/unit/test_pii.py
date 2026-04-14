"""Tests for PII detection and masking.

Covers:
- Aadhaar numbers (spaces, hyphens, no separators)
- Phone numbers (Indian mobile starting with 6-9)
- PAN cards (ABCDE1234F format)
- Mixed text with multiple PII types
- Text without PII (should remain unchanged)
- Edge cases: partial numbers, numbers in context, overlapping patterns
- detect_pii() and contains_aadhaar() functions
"""

from __future__ import annotations

from vaidya.compliance.pii import contains_aadhaar, detect_pii, mask_pii

# ---------------------------------------------------------------------------
# mask_pii: Aadhaar
# ---------------------------------------------------------------------------


class TestMaskPIIAadhaar:
    def test_aadhaar_with_spaces(self) -> None:
        assert mask_pii("My Aadhaar is 1234 5678 9012") == "My Aadhaar is XXXX-XXXX-XXXX"

    def test_aadhaar_with_hyphens(self) -> None:
        assert mask_pii("Aadhaar: 1234-5678-9012") == "Aadhaar: XXXX-XXXX-XXXX"

    def test_aadhaar_no_separators(self) -> None:
        assert mask_pii("Number 123456789012") == "Number XXXX-XXXX-XXXX"

    def test_multiple_aadhaar_in_same_text(self) -> None:
        text = "First: 1234 5678 9012, second: 9876-5432-1098"
        result = mask_pii(text)
        assert "1234" not in result
        assert "9876" not in result
        assert result.count("XXXX-XXXX-XXXX") == 2

    def test_aadhaar_at_start_of_text(self) -> None:
        assert mask_pii("1234 5678 9012 is my Aadhaar") == "XXXX-XXXX-XXXX is my Aadhaar"

    def test_aadhaar_at_end_of_text(self) -> None:
        assert mask_pii("Aadhaar number: 1234-5678-9012") == "Aadhaar number: XXXX-XXXX-XXXX"


# ---------------------------------------------------------------------------
# mask_pii: Phone
# ---------------------------------------------------------------------------


class TestMaskPIIPhone:
    def test_phone_number(self) -> None:
        result = mask_pii("Call me at 9876543210")
        assert "9876543210" not in result
        assert "XXXXXXXXXX" in result

    def test_phone_starting_with_6(self) -> None:
        result = mask_pii("Number: 6123456789")
        assert "6123456789" not in result
        assert "XXXXXXXXXX" in result

    def test_phone_starting_with_7(self) -> None:
        result = mask_pii("Phone 7000111222")
        assert "7000111222" not in result

    def test_phone_starting_with_8(self) -> None:
        result = mask_pii("Dial 8765432109")
        assert "8765432109" not in result

    def test_number_starting_with_5_not_phone(self) -> None:
        """Numbers starting with 5 are not Indian mobile numbers."""
        text = "Code: 5123456789"
        result = mask_pii(text)
        # 5xxx... is not a valid Indian mobile pattern (must start with 6-9)
        # But may be treated as part of 12-digit Aadhaar if context matches
        # The key is it should NOT be masked as a phone
        assert "XXXXXXXXXX" not in result or "5123456789" not in text


# ---------------------------------------------------------------------------
# mask_pii: PAN
# ---------------------------------------------------------------------------


class TestMaskPIIPan:
    def test_pan_card(self) -> None:
        assert "ABCDE1234F" not in mask_pii("PAN: ABCDE1234F")
        assert "XXXXX0000X" in mask_pii("PAN: ABCDE1234F")

    def test_pan_different_letters(self) -> None:
        result = mask_pii("PAN number ZYXWV9876A")
        assert "ZYXWV9876A" not in result
        assert "XXXXX0000X" in result

    def test_lowercase_pan_not_matched(self) -> None:
        """PAN pattern requires uppercase -- lowercase should not match."""
        text = "abcde1234f is not a PAN"
        assert mask_pii(text) == text


# ---------------------------------------------------------------------------
# mask_pii: Mixed and edge cases
# ---------------------------------------------------------------------------


class TestMaskPIIMixed:
    def test_mixed_text_with_aadhaar_and_phone(self) -> None:
        text = "Mera Aadhaar 1234 5678 9012 hai aur phone 9876543210 hai"
        result = mask_pii(text)
        assert "1234" not in result
        assert "9876543210" not in result
        assert "XXXX-XXXX-XXXX" in result
        assert "XXXXXXXXXX" in result

    def test_mixed_text_with_all_pii_types(self) -> None:
        text = "Aadhaar 1234 5678 9012, PAN ABCDE1234F, phone 9876543210"
        result = mask_pii(text)
        assert "1234 5678 9012" not in result
        assert "ABCDE1234F" not in result
        assert "9876543210" not in result

    def test_no_pii_text_unchanged(self) -> None:
        text = "Main Rajasthan se hoon aur daily mazdoori karta hoon"
        assert mask_pii(text) == text

    def test_partial_numbers_unchanged(self) -> None:
        """Short numbers that are not PII should remain."""
        text = "5 log hain ghar mein, income 2.5 lakh"
        assert mask_pii(text) == text

    def test_empty_string(self) -> None:
        assert mask_pii("") == ""

    def test_only_numbers_short(self) -> None:
        """Short digit strings that are not PII."""
        assert mask_pii("12345") == "12345"

    def test_numbers_in_context_not_masked(self) -> None:
        """Common numbers in conversation (age, family size) should not trigger masking."""
        text = "Meri umar 45 saal hai aur 3 bacche hain"
        assert mask_pii(text) == text

    def test_aadhaar_takes_priority_over_phone(self) -> None:
        """A 12-digit Aadhaar should be masked as Aadhaar, not partially as phone."""
        text = "Number: 987654321098"
        result = mask_pii(text)
        # Should be masked as Aadhaar (12 digits) -> XXXX-XXXX-XXXX
        assert "XXXX-XXXX-XXXX" in result

    def test_hindi_text_with_pii(self) -> None:
        text = "मेरा आधार नंबर 1234 5678 9012 है"
        result = mask_pii(text)
        assert "1234" not in result
        assert "XXXX-XXXX-XXXX" in result


# ---------------------------------------------------------------------------
# detect_pii
# ---------------------------------------------------------------------------


class TestDetectPII:
    def test_finds_aadhaar(self) -> None:
        findings = detect_pii("Aadhaar: 1234 5678 9012")
        aadhaar = [f for f in findings if f.pii_type == "aadhaar"]
        assert len(aadhaar) == 1
        assert aadhaar[0].masked_value == "XXXX-XXXX-XXXX"

    def test_finds_phone(self) -> None:
        findings = detect_pii("Phone: 9876543210")
        phones = [f for f in findings if f.pii_type == "phone"]
        assert len(phones) == 1
        assert phones[0].masked_value == "XXXXXXXXXX"

    def test_finds_pan(self) -> None:
        findings = detect_pii("PAN: ABCDE1234F")
        pans = [f for f in findings if f.pii_type == "pan"]
        assert len(pans) == 1
        assert pans[0].masked_value == "XXXXX0000X"

    def test_empty_for_clean_text(self) -> None:
        assert detect_pii("Hello world") == []

    def test_multiple_pii_types(self) -> None:
        findings = detect_pii("Aadhaar 1234 5678 9012, phone 9876543210, PAN ABCDE1234F")
        types = {f.pii_type for f in findings}
        assert "aadhaar" in types
        assert "phone" in types
        assert "pan" in types

    def test_findings_sorted_by_position(self) -> None:
        """Findings should be sorted by start position."""
        findings = detect_pii("PAN ABCDE1234F then Aadhaar 1234 5678 9012")
        if len(findings) >= 2:
            for i in range(len(findings) - 1):
                assert findings[i].start <= findings[i + 1].start

    def test_positions_are_correct(self) -> None:
        text = "Number: 1234 5678 9012"
        findings = detect_pii(text)
        aadhaar = [f for f in findings if f.pii_type == "aadhaar"]
        assert len(aadhaar) == 1
        # The matched span should cover the Aadhaar portion
        assert aadhaar[0].start >= 0
        assert aadhaar[0].end <= len(text)

    def test_phone_not_detected_inside_aadhaar(self) -> None:
        """A 10-digit substring inside a 12-digit Aadhaar should not be detected as phone."""
        findings = detect_pii("1234 9876 5432")
        phones = [f for f in findings if f.pii_type == "phone"]
        # Should not find a phone inside the Aadhaar span
        assert len(phones) == 0


# ---------------------------------------------------------------------------
# contains_aadhaar
# ---------------------------------------------------------------------------


class TestContainsAadhaar:
    def test_true_for_aadhaar_with_spaces(self) -> None:
        assert contains_aadhaar("1234 5678 9012") is True

    def test_true_for_aadhaar_with_hyphens(self) -> None:
        assert contains_aadhaar("1234-5678-9012") is True

    def test_true_for_aadhaar_no_separators(self) -> None:
        assert contains_aadhaar("123456789012") is True

    def test_false_for_no_aadhaar(self) -> None:
        assert contains_aadhaar("No numbers here") is False

    def test_false_for_short_number(self) -> None:
        assert contains_aadhaar("12345678") is False

    def test_true_for_aadhaar_in_sentence(self) -> None:
        assert contains_aadhaar("My number is 1234 5678 9012 please note") is True

    def test_false_for_empty_string(self) -> None:
        assert contains_aadhaar("") is False
