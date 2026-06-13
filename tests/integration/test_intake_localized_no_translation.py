"""Intake runs in the caller's language with NO redundant translation hops.

The latency win: an intake turn used to make THREE sequential Sarvam calls --
translate-in (user-lang -> en-IN), LLM extract, translate-out (en-IN ->
user-lang). The two translation hops are redundant: sarvam-30b reads the
caller's language directly and still emits canonical English field values.

These tests lock in the new contract:

1. The IntakeAgent marks every response ``already_localized=True`` (so the
   ConversationManager skips translate-out -- the hop where the
   placeholder-leak / blank-turn bugs lived).
2. Field VALUES stay canonical English enums (``state``, ``occupation_type``,
   ``income_bracket``, ``existing_coverage``) even though the caller spoke a
   non-English language -- the eligibility/reviewer/RAG pipeline depends on
   this and must not change.
3. The ConversationManager makes ZERO translate calls on an intake-phase turn
   (translate-in skipped because the phase is intake-bound; translate-out
   skipped because the response is ``already_localized``), while a non-intake
   en-IN turn still translates as before.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from vaidya.agents.intake import IntakeAgent
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.user_profile import (
    CoverageType,
    IncomeCategory,
    OccupationType,
)
from vaidya.pipeline.conversation import ConversationManager

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _IntakeMockClient:
    """A SarvamClient stub that mimics sarvam-30b reading a NON-English answer.

    The canned response emits canonical English field VALUES (the safety
    invariant) while the spoken ack is in the caller's language -- exactly the
    behaviour the updated intake prompt asks for. ``translate`` raises so any
    accidental translate call on the intake path fails the test loudly.
    """

    def __init__(self, fields: dict[str, Any] | None = None) -> None:
        self.translate_calls = 0
        self._fields = fields or {
            "state": "Maharashtra",
            "district": "Pune",
            "family_size": 4,
            "occupation_type": "daily_wage",
            "income_bracket": "below_1l",
            "existing_coverage": "none",
            "health_need": "knee operation",
        }

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        return json.dumps(
            {
                # Spoken ack is in the caller's language (Marathi here).
                "spoken_text": "Bara, samajla.",
                "extracted_fields": dict(self._fields),
                "field_confidence": {k: 0.9 for k in self._fields},
                "question_complete": True,
                "needs_followup": False,
                "distress_detected": False,
            }
        )

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raw = await self.chat(model, messages, temperature)
        return json.loads(raw)

    async def translate(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        self.translate_calls += 1
        raise AssertionError("translate() must not be called on the intake path")


def _make_context(
    phase: ConversationPhase = ConversationPhase.INTAKE,
    language: str = "mr-IN",
    q_index: int = 1,
) -> ConversationContext:
    ctx = ConversationContext(
        call_id="test-intake-localized",
        phone_number_hash="hash_abc",
        language=language,
        phase=phase,
    )
    ctx.intake_question_index = q_index
    return ctx


# ===========================================================================
# IntakeAgent: already_localized + canonical extraction
# ===========================================================================


class TestIntakeAlreadyLocalized:
    """Every IntakeAgent response is flagged already_localized=True."""

    async def test_question_answer_response_is_already_localized(self) -> None:
        agent = IntakeAgent(_IntakeMockClient(), model="mock-model")
        ctx = _make_context(q_index=1)

        # Caller answers Q1 in Marathi (raw, untranslated).
        response = await agent.process(ctx, "Mi Maharashtra madhe rahato.")

        assert response.already_localized is True

    async def test_initial_freeform_response_is_already_localized(self) -> None:
        agent = IntakeAgent(_IntakeMockClient(), model="mock-model")
        ctx = _make_context(q_index=0)

        response = await agent.process(ctx, "Mala arogya yojana havi aahe.")

        assert response.already_localized is True

    async def test_confirmation_response_is_already_localized(self) -> None:
        agent = IntakeAgent(_IntakeMockClient(), model="mock-model")
        ctx = _make_context(q_index=6)
        ctx.metadata["confirmation_pending"] = True

        response = await agent.process(ctx, "Ho, barobar aahe.")

        assert response.already_localized is True

    async def test_empty_first_turn_response_is_already_localized(self) -> None:
        agent = IntakeAgent(_IntakeMockClient(), model="mock-model")
        ctx = _make_context(q_index=0)

        response = await agent.process(ctx, "")

        assert response.already_localized is True

    async def test_error_fallback_is_already_localized(self) -> None:
        """safe_process error fallback is a localized i18n string -- flag it.

        Force ``process`` to raise so ``BaseAgent.safe_process`` returns its
        ``_fallback_response`` (a ``get_msg(..., language)`` string). The
        IntakeAgent override must still stamp ``already_localized=True`` so the
        manager never translates the already-localized fallback back out.
        """
        agent = IntakeAgent(_IntakeMockClient(), model="mock-model")
        ctx = _make_context(q_index=1)

        async def _boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("LLM down")

        agent._process_turn = _boom  # type: ignore[method-assign]

        response = await agent.safe_process(ctx, "Mi Maharashtra madhe rahato.")
        assert response.error == "agent_processing_failed"
        assert response.already_localized is True


class TestCanonicalExtractionUnchanged:
    """Field VALUES stay canonical English even for a non-English caller."""

    async def test_canonical_field_values_from_marathi_input(self) -> None:
        agent = IntakeAgent(_IntakeMockClient(), model="mock-model")
        ctx = _make_context(q_index=0, language="mr-IN")

        # A single rich Marathi free-form statement; the mock returns the full
        # set of canonical fields. The agent applies them to the returned
        # ``updated_profile`` (the orchestrator is what writes it back to ctx).
        response = await agent.process(ctx, "Mi Maharashtra, Pune. Rojanderi kaam. Vima nahi.")

        profile = response.updated_profile
        assert profile is not None
        # Canonical state name (English), not a Marathi rendering.
        assert profile.state == "Maharashtra"
        # Canonical enums -- the safety-critical invariant for eligibility/RAG.
        assert profile.occupation_type == OccupationType.DAILY_WAGE
        assert profile.income_bracket == IncomeCategory.BELOW_1L
        assert profile.existing_coverage == CoverageType.NONE
        assert profile.family_size == 4

    async def test_state_code_canonicalized_when_llm_emits_code(self) -> None:
        """An LLM that emits a canonical code/value is applied verbatim."""
        client = _IntakeMockClient(
            fields={
                "state": "Maharashtra",
                "occupation_type": "farmer",
                "income_bracket": "1l_to_2.5l",
                "existing_coverage": "govt_scheme",
            }
        )
        agent = IntakeAgent(client, model="mock-model")
        ctx = _make_context(q_index=0, language="mr-IN")

        response = await agent.process(ctx, "Mi shetkari aahe.")

        profile = response.updated_profile
        assert profile is not None
        assert profile.occupation_type == OccupationType.FARMER
        assert profile.income_bracket == IncomeCategory.L1_TO_2_5L
        assert profile.existing_coverage == CoverageType.GOVT_SCHEME


# ===========================================================================
# ConversationManager: no translation on the intake path
# ===========================================================================


def _build_manager(
    response,
    session_ctx: ConversationContext,
) -> tuple[ConversationManager, MagicMock]:
    """Wire a ConversationManager with a spy translator and a stub orchestrator."""
    orchestrator = MagicMock()
    orchestrator.handle_turn = AsyncMock(return_value=response)

    session = MagicMock()
    session.get = AsyncMock(return_value=session_ctx)
    session.update = AsyncMock()

    translator = MagicMock()
    translator.translate_if_needed = AsyncMock(side_effect=lambda text, *a, **k: text)

    audit = MagicMock()
    audit.log_event = MagicMock()
    audit.log_turn = MagicMock()
    audit.log_eligibility_decision = MagicMock()

    mgr = ConversationManager(
        orchestrator=orchestrator,
        session_manager=session,
        translator=translator,
        audit_trail=audit,
    )
    return mgr, translator


class TestNoTranslationOnIntakePath:
    """The manager makes ZERO translate calls on an intake-phase turn."""

    async def test_intake_turn_skips_translate_in_and_out(self) -> None:
        from vaidya.models.api import AgentResponse

        ctx = ConversationContext(
            call_id="c1",
            phone_number_hash="h",
            language="mr-IN",
            phase=ConversationPhase.INTAKE,
        )
        # Mirror what the real intake agent returns: localized + flagged.
        response = AgentResponse(
            text="Bara. Tumchya gharat kiti manse aahet?",
            already_localized=True,
            metadata={"agent": "intake", "intake_q": 2},
        )
        mgr, translator = _build_manager(response, ctx)

        out = await mgr.handle_turn("c1", "Mi Maharashtra madhe rahato.")

        # No translate-in (intake-bound phase) and no translate-out
        # (already_localized) -- both redundant hops eliminated.
        translator.translate_if_needed.assert_not_awaited()
        # The raw caller utterance reaches the orchestrator untranslated.
        assert mgr._orchestrator.handle_turn.await_args.args[1] == "Mi Maharashtra madhe rahato."
        # The localized text is returned verbatim.
        assert out == "Bara. Tumchya gharat kiti manse aahet?"

    async def test_open_elicitation_turn_also_skips_translate_in(self) -> None:
        from vaidya.models.api import AgentResponse

        ctx = ConversationContext(
            call_id="c2",
            phone_number_hash="h",
            language="mr-IN",
            phase=ConversationPhase.OPEN_ELICITATION,
        )
        response = AgentResponse(
            text="Bara. Tumhi kuthe rahata?",
            already_localized=True,
            metadata={"agent": "intake"},
        )
        mgr, translator = _build_manager(response, ctx)

        await mgr.handle_turn("c2", "Mala madat havi aahe.")

        translator.translate_if_needed.assert_not_awaited()
        assert mgr._orchestrator.handle_turn.await_args.args[1] == "Mala madat havi aahe."

    async def test_non_intake_turn_still_translates_in(self) -> None:
        """A GUIDANCE-phase turn keeps the en-IN path: translate-in still runs."""
        from vaidya.models.api import AgentResponse

        ctx = ConversationContext(
            call_id="c3",
            phone_number_hash="h",
            language="mr-IN",
            phase=ConversationPhase.GUIDANCE,
        )
        # Guidance response is NOT flagged already_localized in this stub, so
        # translate-out would also run -- proving the en-IN path is intact.
        response = AgentResponse(
            text="Visit your nearest Jan Seva Kendra.",
            already_localized=False,
            metadata={"agent": "guidance"},
        )
        mgr, translator = _build_manager(response, ctx)

        await mgr.handle_turn("c3", "PMJAY baddal sanga.")

        # translate-in (mr-IN -> en-IN) AND translate-out (en-IN -> mr-IN).
        assert translator.translate_if_needed.await_count == 2
        first_call = translator.translate_if_needed.await_args_list[0]
        assert first_call.args == ("PMJAY baddal sanga.", "mr-IN", "en-IN")

    async def test_english_intake_turn_makes_no_translate_call(self) -> None:
        """An English caller in intake also makes zero translate calls."""
        from vaidya.models.api import AgentResponse

        ctx = ConversationContext(
            call_id="c4",
            phone_number_hash="h",
            language="en-IN",
            phase=ConversationPhase.INTAKE,
        )
        response = AgentResponse(
            text="Got it. How many people live in your home?",
            already_localized=True,
            metadata={"agent": "intake", "intake_q": 2},
        )
        mgr, translator = _build_manager(response, ctx)

        await mgr.handle_turn("c4", "I live in Maharashtra.")

        translator.translate_if_needed.assert_not_awaited()
