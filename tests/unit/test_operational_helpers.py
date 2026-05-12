"""Focused coverage for operational helper modules."""

from __future__ import annotations

import pytest

from vaidya.models.user_profile import UserProfile
from vaidya.pipeline.degradation import DegradationLevel, DegradationManager
from vaidya.utils.state_metadata import STATE_METADATA, get_primary_language, get_state_info
from vaidya.validation import ValidationError, validate_language, validate_state
from vaidya.voice.document import DocumentVerifier, VerificationResult


class TestStateMetadata:
    def test_state_metadata_contains_all_known_states_and_flags_opt_outs(self):
        assert len(STATE_METADATA) == 36
        assert get_state_info(" wb ").pmjay_excluded is True
        assert get_state_info("TN").scheme_ids == ["CMCHIS-TN-2024-v1"]
        assert get_state_info("xx") is None

    def test_primary_language_falls_back_to_hindi_for_unknown_state(self):
        assert get_primary_language("TN") == "ta-IN"
        assert get_primary_language("XX") == "hi-IN"


class TestDegradationManager:
    def test_tracks_highest_degradation_level_and_failed_services(self):
        manager = DegradationManager(threshold=2)

        manager.record_failure("reviewer")
        assert manager.level == DegradationLevel.FULL
        assert manager.is_service_available("reviewer") is True

        manager.record_failure("reviewer")
        manager.record_failure("voice")
        manager.record_failure("voice")

        assert manager.level == DegradationLevel.SMS_ONLY
        assert manager.failed_services == ["reviewer", "voice"]
        assert manager.is_service_available("voice") is False

    def test_record_success_recovers_service(self):
        manager = DegradationManager(threshold=1)
        manager.record_failure("translator")

        assert manager.level == DegradationLevel.REDUCED_LANGUAGES

        manager.record_success("translator")

        assert manager.level == DegradationLevel.FULL
        assert manager.failed_services == []


class TestDocumentVerifier:
    async def test_verify_document_returns_placeholder_result(self):
        verifier = DocumentVerifier(client=object())
        result = await verifier.verify_document(
            image_file=object(),
            user_profile=UserProfile(state="Rajasthan"),
        )

        assert isinstance(result, VerificationResult)
        assert result.extracted_fields == {}
        assert result.masked_text == ""
        assert result.discrepancies == []
        assert result.verified is True

    def test_cross_check_reports_state_mismatch(self):
        verifier = DocumentVerifier(client=object())

        issues = verifier._cross_check(
            {"state": "Tamil Nadu", "name": "Example Person"},
            UserProfile(state="Rajasthan"),
        )

        assert "State mismatch" in issues[0]


class TestValidation:
    def test_voice_language_accepts_supported_voice_language(self):
        assert validate_language("ta", channel="voice") == "ta-IN"

    def test_text_only_language_rejected_for_voice_but_allowed_for_text(self):
        with pytest.raises(ValidationError, match="only supported on text channels"):
            validate_language("ur-IN", channel="voice")

        assert validate_language("ur-IN", channel="sms") == "ur-IN"

    def test_unknown_language_and_state_raise_clear_errors(self):
        with pytest.raises(ValidationError, match="not supported"):
            validate_language("klingon")
        with pytest.raises(ValidationError, match="not recognized"):
            validate_state("Atlantis")

    def test_validate_state_returns_state_code(self):
        assert validate_state("West Bengal") == "WB"
