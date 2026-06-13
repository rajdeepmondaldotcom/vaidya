"""Fast-path intake: skip the LLM for short, clear answers; fall back otherwise.

The fast path reuses the same deterministic ``_heuristic_fields`` extraction that
already runs as a backup on every turn, so extracted VALUES are unchanged — the
only new behaviour is skipping the LLM call when the heuristics confidently cover
the question's required field(s). A mis-read is caught by the confirmation step.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from vaidya.agents.intake import IntakeAgent
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.user_profile import UserProfile


def _agent() -> IntakeAgent:
    return IntakeAgent(client=object(), model="mock")


class TestTryFastExtractGate:
    def test_state_clear_answer_takes_fast_path(self):
        fast = _agent()._try_fast_extract("Main Rajasthan mein rehta hoon", 1, "hi-IN")
        assert fast is not None
        assert fast["extracted_fields"]["state"] == "Rajasthan"
        assert fast["question_complete"] is True
        assert fast["spoken_text"] == "Theek hai."  # hi-IN ack

    def test_family_size_fast_path(self):
        fast = _agent()._try_fast_extract("hamare ghar mein chaar log hain", 2, "hi-IN")
        assert fast is not None
        assert fast["extracted_fields"]["family_size"] == 4

    def test_occupation_and_income_fast_path(self):
        fast = _agent()._try_fast_extract("daily wage majdoori, mahine 7 hazaar", 3, "hi-IN")
        assert fast is not None
        assert fast["extracted_fields"]["occupation_type"] == "daily_wage"
        assert fast["extracted_fields"]["income_bracket"] == "below_1l"

    def test_coverage_negation_is_the_answer(self):
        # On q4 a "no" IS the answer (none) — handled by the coverage heuristic.
        fast = _agent()._try_fast_extract("nahi, koi insurance nahi hai", 4, "hi-IN")
        assert fast is not None
        assert fast["extracted_fields"]["existing_coverage"] == "none"

    def test_negated_state_falls_back_to_llm(self):
        # "Bihar nahi, Jharkhand" — negation guard must force the LLM (q1).
        assert _agent()._try_fast_extract("Bihar nahi, Jharkhand mein", 1, "hi-IN") is None

    def test_incomplete_q3_falls_back(self):
        # Occupation present but no income → required fields not fully covered.
        assert _agent()._try_fast_extract("main farmer hoon", 3, "hi-IN") is None

    def test_long_answer_falls_back(self):
        long_answer = (
            "main pehle Bihar mein tha lekin ab kaam ke liye Rajasthan aa gaya "
            "hoon aur abhi yahin rehta hoon apne parivaar ke saath"
        )
        assert _agent()._try_fast_extract(long_answer, 1, "hi-IN") is None

    def test_health_need_always_uses_llm(self):
        assert _agent()._try_fast_extract("dil ki bimari ka ilaaj", 5, "hi-IN") is None

    def test_initial_freeform_always_uses_llm(self):
        assert _agent()._try_fast_extract("Rajasthan, 4 log, daily wage", 0, "hi-IN") is None

    def test_unknown_language_never_speaks_the_key(self):
        fast = _agent()._try_fast_extract("Rajasthan mein rehta hoon", 1, "zz-ZZ")
        assert fast is not None
        assert fast["spoken_text"] != "ack"  # defensive: never leak the i18n key


@pytest.mark.asyncio
class TestHandleQuestionAnswerSkipsLLM:
    @staticmethod
    def _ctx() -> ConversationContext:
        return ConversationContext(
            call_id="t",
            phone_number_hash="h",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )

    async def test_clear_answer_skips_the_llm(self):
        agent = _agent()
        agent._extract_answer = AsyncMock()  # must NOT be awaited on the fast path
        resp = await agent._handle_question_answer(
            self._ctx(), UserProfile(), "Main Rajasthan mein rehta hoon", 1, "hi-IN"
        )
        agent._extract_answer.assert_not_awaited()
        assert resp.updated_profile.state == "Rajasthan"  # canonical value preserved

    async def test_health_need_falls_back_to_llm(self):
        agent = _agent()
        agent._extract_answer = AsyncMock(
            return_value={"extracted_fields": {"health_need": "heart"}, "field_confidence": {}}
        )
        await agent._handle_question_answer(
            self._ctx(), UserProfile(), "mujhe dil ki bimari hai", 5, "hi-IN"
        )
        agent._extract_answer.assert_awaited_once()  # q5 is free-form → always LLM
