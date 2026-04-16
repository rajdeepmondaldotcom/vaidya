"""Tests for prompt template registry and template content.

Covers:
- Each of the 5 templates loads successfully
- Templates contain required domain-specific keywords
- render() fills placeholders correctly
- get_raw() returns unrendered template text
- Missing template raises FileNotFoundError
- Cache clearing
"""

from __future__ import annotations

import pytest

from vaidya.prompts.registry import clear_cache, get_raw, render

# All expected template names
_TEMPLATE_NAMES = [
    "intake_system",
    "eligibility_system",
    "guidance_system",
    "reviewer_system",
    "orchestrator_fallback",
]


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


class TestTemplateLoading:
    def test_intake_system_loads(self) -> None:
        text = get_raw("intake_system")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_eligibility_system_loads(self) -> None:
        text = get_raw("eligibility_system")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_guidance_system_loads(self) -> None:
        text = get_raw("guidance_system")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_reviewer_system_loads(self) -> None:
        text = get_raw("reviewer_system")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_orchestrator_fallback_loads(self) -> None:
        text = get_raw("orchestrator_fallback")
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Template keyword checks
# ---------------------------------------------------------------------------


class TestTemplateKeywords:
    def test_intake_mentions_json(self) -> None:
        text = get_raw("intake_system")
        assert "JSON" in text

    def test_eligibility_mentions_scheme_id(self) -> None:
        text = get_raw("eligibility_system")
        assert "scheme_id" in text

    def test_guidance_mentions_spoken_parts(self) -> None:
        text = get_raw("guidance_system")
        assert "spoken_parts" in text

    def test_reviewer_mentions_transcript(self) -> None:
        text = get_raw("reviewer_system")
        assert "transcript" in text.lower()

    def test_orchestrator_fallback_mentions_phase(self) -> None:
        text = get_raw("orchestrator_fallback")
        assert "phase" in text.lower()


# ---------------------------------------------------------------------------
# render() fills placeholders
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_orchestrator_fallback_fills_placeholders(self) -> None:
        result = render(
            "orchestrator_fallback",
            phase="INTAKE",
            language="hi-IN",
            recent_turns="User: Namaste",
            user_input="Main Jaipur se hoon",
        )
        assert "INTAKE" in result
        assert "hi-IN" in result
        assert "Main Jaipur se hoon" in result
        # Placeholders should have been replaced -- no bare {phase}
        assert "{phase}" not in result

    def test_render_intake_fills_question_number(self) -> None:
        result = render(
            "intake_system",
            question_number="1",
            current_question="Aap kahan se bol rahe hain?",
            profile_summary="None collected yet",
            language="hi-IN",
            expected_fields_json='{"state": null}',
        )
        assert "1" in result
        assert "{question_number}" not in result


# ---------------------------------------------------------------------------
# get_raw() returns unrendered template
# ---------------------------------------------------------------------------


class TestGetRaw:
    def test_get_raw_returns_unrendered_text_with_placeholders(self) -> None:
        raw = get_raw("orchestrator_fallback")
        # The raw template should contain format-string placeholders
        assert "{phase}" in raw or "{language}" in raw


# ---------------------------------------------------------------------------
# Missing template
# ---------------------------------------------------------------------------


class TestMissingTemplate:
    def test_missing_template_raises_file_not_found_error(self) -> None:
        clear_cache()  # ensure we hit filesystem
        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            get_raw("nonexistent_template")


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestCacheBehaviour:
    def test_clear_cache_forces_reload(self) -> None:
        # Load a template to populate cache
        text1 = get_raw("intake_system")
        clear_cache()
        # Reloading after cache clear should still work
        text2 = get_raw("intake_system")
        assert text1 == text2
