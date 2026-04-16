"""Tests for the i18n message system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaidya.i18n.messages import get_msg, get_msg_template, list_keys, reload

_STRINGS_DIR = Path(__file__).resolve().parents[2] / "src" / "vaidya" / "i18n" / "strings"

# All domains that must exist
_DOMAINS = ["orchestrator", "intake", "guidance", "conversation", "base_agent"]

# Languages that EVERY key must have (minimum coverage)
_REQUIRED_LANGS = {"hi-IN", "en-IN"}

# All 11 TTS voice languages
_ALL_VOICE_LANGS = {
    "hi-IN",
    "ta-IN",
    "bn-IN",
    "te-IN",
    "gu-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "od-IN",
    "en-IN",
}


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the i18n cache before each test."""
    reload()


class TestDomainFilesExist:
    """Verify all expected JSON domain files exist and are valid JSON."""

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_domain_file_exists(self, domain: str) -> None:
        path = _STRINGS_DIR / f"{domain}.json"
        assert path.exists(), f"Missing i18n domain file: {path}"

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_domain_file_valid_json(self, domain: str) -> None:
        path = _STRINGS_DIR / f"{domain}.json"
        with path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) > 0


class TestMinimumLanguageCoverage:
    """Every key in every domain must have at least hi-IN and en-IN."""

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_required_langs_present(self, domain: str) -> None:
        path = _STRINGS_DIR / f"{domain}.json"
        with path.open() as f:
            data = json.load(f)

        missing = []
        for key, translations in data.items():
            for lang in _REQUIRED_LANGS:
                if lang not in translations:
                    missing.append(f"{domain}.{key} missing {lang}")

        assert not missing, "Missing required translations:\n" + "\n".join(missing)


class TestFullVoiceLanguageCoverage:
    """Every key should have all 11 voice languages."""

    @pytest.mark.parametrize("domain", _DOMAINS)
    def test_all_voice_langs_present(self, domain: str) -> None:
        path = _STRINGS_DIR / f"{domain}.json"
        with path.open() as f:
            data = json.load(f)

        missing = []
        for key, translations in data.items():
            for lang in _ALL_VOICE_LANGS:
                if lang not in translations:
                    missing.append(f"{domain}.{key} missing {lang}")

        assert not missing, "Missing voice language translations:\n" + "\n".join(missing)


class TestGetMsg:
    """Test the get_msg accessor function."""

    def test_exact_match(self) -> None:
        msg = get_msg("orchestrator", "welcome", "hi-IN")
        assert "Namaste" in msg
        assert "Vaidya" in msg

    def test_english_match(self) -> None:
        msg = get_msg("orchestrator", "welcome", "en-IN")
        assert "Hello" in msg

    def test_fallback_to_hindi(self) -> None:
        # Request a non-existent language — should fall back to hi-IN
        msg = get_msg("orchestrator", "welcome", "xx-XX")
        assert "Namaste" in msg

    def test_missing_key_returns_key(self) -> None:
        msg = get_msg("orchestrator", "nonexistent_key", "hi-IN")
        assert msg == "nonexistent_key"

    def test_missing_domain_returns_key(self) -> None:
        msg = get_msg("nonexistent_domain", "welcome", "hi-IN")
        assert msg == "welcome"


class TestGetMsgTemplate:
    """Test template formatting."""

    def test_state_template(self) -> None:
        msg = get_msg_template("intake", "confirm_state", "hi-IN", state="Maharashtra")
        assert "Maharashtra" in msg
        assert "rehte hain" in msg

    def test_family_template(self) -> None:
        msg = get_msg_template("intake", "confirm_family", "en-IN", count=5)
        assert "5" in msg

    def test_scheme_name_template(self) -> None:
        msg = get_msg_template("guidance", "fallback_headline", "hi-IN", scheme_name="PM-JAY")
        assert "PM-JAY" in msg

    def test_missing_placeholder_returns_template(self) -> None:
        # Missing kwargs should return template as-is (no crash)
        msg = get_msg_template("intake", "confirm_state", "hi-IN")
        assert "{state}" in msg


class TestIntakeMessages:
    """Verify intake-specific message structure."""

    def test_all_five_questions_exist(self) -> None:
        keys = list_keys("intake")
        for q in range(1, 6):
            assert f"q{q}" in keys, f"Missing question q{q}"

    def test_occupation_labels_exist(self) -> None:
        keys = list_keys("intake")
        for occ in ["daily_wage", "salaried_govt", "salaried_pvt", "self_employed", "farmer"]:
            assert f"occ_{occ}" in keys, f"Missing occupation label occ_{occ}"

    def test_confirmation_templates_exist(self) -> None:
        keys = list_keys("intake")
        for part in ["confirm_prefix", "confirm_state", "confirm_family", "confirm_suffix"]:
            assert part in keys, f"Missing confirmation template {part}"


class TestOrchestratorMessages:
    """Verify orchestrator-specific message structure."""

    def test_silence_handlers_exist(self) -> None:
        keys = list_keys("orchestrator")
        for sec in [5, 10, 15]:
            assert f"silence_{sec}s" in keys, f"Missing silence handler silence_{sec}s"

    def test_intent_word_lists_exist(self) -> None:
        keys = list_keys("orchestrator")
        for intent in ["negative_words", "restart_words", "continue_words"]:
            assert intent in keys, f"Missing intent word list {intent}"

    def test_intent_words_are_comma_separated(self) -> None:
        words = get_msg("orchestrator", "negative_words", "hi-IN")
        parts = words.split(",")
        assert len(parts) >= 2, "Intent word list should be comma-separated"


class TestGuidanceMessages:
    """Verify guidance-specific message structure."""

    def test_no_match_exists(self) -> None:
        msg = get_msg("guidance", "no_match", "hi-IN")
        assert "yojana" in msg.lower() or "Jan Seva Kendra" in msg

    def test_fallback_sms_template(self) -> None:
        msg = get_msg_template("guidance", "fallback_sms", "hi-IN", names="PM-JAY")
        assert "PM-JAY" in msg
        assert "Vaidya" in msg
