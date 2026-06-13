"""Unit tests for the ConversationManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase, Turn
from vaidya.pipeline.conversation import ConversationManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    call_id: str = "test-call-001",
    language: str = "hi-IN",
    phase: ConversationPhase = ConversationPhase.WELCOME,
) -> ConversationContext:
    return ConversationContext(
        call_id=call_id,
        phone_number_hash="hash123",
        language=language,
        phase=phase,
    )


def _make_deps(
    orchestrator_text: str = "Namaste! Main Vaidya hoon.",
    session_context: ConversationContext | None = None,
):
    """Build mocked dependencies for ConversationManager."""
    orchestrator = MagicMock()
    orchestrator.handle_turn = AsyncMock(
        return_value=AgentResponse(text=orchestrator_text, metadata={"agent": "orchestrator"})
    )

    session_manager = MagicMock()
    session_manager.generate_call_id = MagicMock(return_value="test-call-001")
    session_manager.find_by_phone = AsyncMock(return_value=None)
    session_manager.create = AsyncMock(return_value=session_context or _make_context())
    session_manager.get = AsyncMock(return_value=session_context or _make_context())
    session_manager.update = AsyncMock()
    session_manager.delete = AsyncMock()

    translator = MagicMock()
    translator.translate_if_needed = AsyncMock(side_effect=lambda text, *a, **kw: text)

    audit = MagicMock()
    audit.log_event = MagicMock()
    audit.log_turn = MagicMock()
    audit.log_eligibility_decision = MagicMock()

    consent = MagicMock()
    consent.record_consent = MagicMock()

    return orchestrator, session_manager, translator, audit, consent


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_accepts_all_parameters(self):
        orch, session, translator, audit, consent = _make_deps()

        def custom_masker(text: str) -> str:
            return text.replace("secret", "***")

        sarvam = MagicMock()

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
            consent_tracker=consent,
            pii_masker=custom_masker,
            sarvam_client=sarvam,
        )
        assert mgr._orchestrator is orch
        assert mgr._session is session
        assert mgr._mask_pii is custom_masker
        assert mgr._sarvam_client is sarvam

    def test_default_pii_masker_used_when_none(self):
        orch, session, translator, audit, consent = _make_deps()
        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        from vaidya.compliance.pii import mask_pii

        assert mgr._mask_pii is mask_pii


# ---------------------------------------------------------------------------
# _default_welcome
# ---------------------------------------------------------------------------


class TestDefaultWelcome:
    def test_hindi(self):
        msg = ConversationManager._default_welcome("hi-IN")
        assert "Vaidya" in msg

    def test_tamil(self):
        msg = ConversationManager._default_welcome("ta-IN")
        assert "Vaidya" in msg

    def test_bengali(self):
        msg = ConversationManager._default_welcome("bn-IN")
        assert "Vaidya" in msg

    def test_english(self):
        msg = ConversationManager._default_welcome("en-IN")
        assert "Vaidya" in msg

    def test_unknown_language_falls_back_to_hindi(self):
        msg = ConversationManager._default_welcome("xx-XX")
        assert msg == ConversationManager._default_welcome("hi-IN")

    def test_kannada_has_own_translation(self):
        msg = ConversationManager._default_welcome("kn-IN")
        assert "Vaidya" in msg


# ---------------------------------------------------------------------------
# _session_expired_message
# ---------------------------------------------------------------------------


class TestSessionExpiredMessage:
    def test_hindi(self):
        msg = ConversationManager._session_expired_message("hi-IN")
        assert "session" in msg.lower()

    def test_tamil(self):
        msg = ConversationManager._session_expired_message("ta-IN")
        assert "session" in msg.lower()

    def test_bengali(self):
        msg = ConversationManager._session_expired_message("bn-IN")
        assert "session" in msg.lower()

    def test_english(self):
        msg = ConversationManager._session_expired_message("en-IN")
        assert "expired" in msg.lower()

    def test_unknown_language_falls_back(self):
        msg = ConversationManager._session_expired_message("xx-XX")
        assert msg == ConversationManager._session_expired_message("hi-IN")


# ---------------------------------------------------------------------------
# _error_message
# ---------------------------------------------------------------------------


class TestErrorMessage:
    def test_hindi(self):
        msg = ConversationManager._error_message("hi-IN")
        assert len(msg) > 0

    def test_tamil(self):
        msg = ConversationManager._error_message("ta-IN")
        assert len(msg) > 0

    def test_bengali(self):
        msg = ConversationManager._error_message("bn-IN")
        assert len(msg) > 0

    def test_english(self):
        msg = ConversationManager._error_message("en-IN")
        assert "sorry" in msg.lower()

    def test_unknown_language_falls_back(self):
        msg = ConversationManager._error_message("xx-XX")
        assert msg == ConversationManager._error_message("hi-IN")


# ---------------------------------------------------------------------------
# Custom PII masker
# ---------------------------------------------------------------------------


class TestCustomPiiMasker:
    @pytest.mark.asyncio
    async def test_custom_masker_is_used(self):
        called_with: list[str] = []

        def tracker_masker(text: str) -> str:
            called_with.append(text)
            return text.upper()

        orch, session, translator, audit, consent = _make_deps()
        ctx = _make_context(phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
            pii_masker=tracker_masker,
        )

        await mgr.handle_turn("test-call-001", "my aadhaar is 1234 5678 9012")
        assert len(called_with) > 0
        assert "my aadhaar is 1234 5678 9012" in called_with


# ---------------------------------------------------------------------------
# handle_turn — session not found
# ---------------------------------------------------------------------------


class TestHandleTurnSessionNotFound:
    @pytest.mark.asyncio
    async def test_returns_expired_message(self):
        orch, session, translator, audit, _ = _make_deps()
        session.get = AsyncMock(return_value=None)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.handle_turn("nonexistent-call", "hello")
        assert "session" in result.lower()


# ---------------------------------------------------------------------------
# handle_turn — orchestrator error
# ---------------------------------------------------------------------------


class TestHandleTurnOrchestratorError:
    @pytest.mark.asyncio
    async def test_returns_error_message_on_failure(self):
        orch, session, translator, audit, _ = _make_deps()
        orch.handle_turn = AsyncMock(side_effect=RuntimeError("LLM down"))
        ctx = _make_context(phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.handle_turn("test-call-001", "hello")
        assert len(result) > 0
        audit.log_turn.assert_called_once()


# ---------------------------------------------------------------------------
# end_conversation
# ---------------------------------------------------------------------------


class TestEndConversation:
    @pytest.mark.asyncio
    async def test_end_conversation_deletes_session(self):
        orch, session, translator, audit, _ = _make_deps()
        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        await mgr.end_conversation("test-call-001")
        session.delete.assert_awaited_once_with("test-call-001")
        audit.log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_conversation_cleans_up_turn_lock(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)
        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        # Create a lock by calling handle_turn
        await mgr.handle_turn("test-call-001", "hello")
        assert "test-call-001" in mgr._turn_locks
        # End should clean it up
        await mgr.end_conversation("test-call-001")
        assert "test-call-001" not in mgr._turn_locks

    @pytest.mark.asyncio
    async def test_mark_voice_disconnected_preserves_session_for_recovery(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)
        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        await mgr.handle_turn("test-call-001", "hello")
        assert "test-call-001" in mgr._turn_locks

        await mgr.mark_voice_disconnected("test-call-001")

        session.delete.assert_not_awaited()
        audit.log_event.assert_any_call("test-call-001", "voice_stream_disconnected")
        assert "test-call-001" not in mgr._turn_locks


# ---------------------------------------------------------------------------
# Turn locking
# ---------------------------------------------------------------------------


class TestTurnLocking:
    @pytest.mark.asyncio
    async def test_turn_creates_lock_per_call_id(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)
        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        await mgr.handle_turn("call-A", "hello")
        await mgr.handle_turn("call-B", "hello")
        assert "call-A" in mgr._turn_locks
        assert "call-B" in mgr._turn_locks
        # Different locks for different calls
        assert mgr._turn_locks["call-A"] is not mgr._turn_locks["call-B"]

    @pytest.mark.asyncio
    async def test_same_call_id_reuses_lock(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)
        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        await mgr.handle_turn("call-A", "hello")
        lock1 = mgr._turn_locks["call-A"]
        await mgr.handle_turn("call-A", "world")
        lock2 = mgr._turn_locks["call-A"]
        assert lock1 is lock2


# ---------------------------------------------------------------------------
# handle_silence (voice edge)
# ---------------------------------------------------------------------------


class TestHandleSilence:
    @pytest.mark.asyncio
    async def test_10s_returns_nudge_not_terminal(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, terminal = await mgr.handle_silence("test-call-001", 10.0)
        assert "yahin hoon" in spoken
        assert terminal is False

    @pytest.mark.asyncio
    async def test_20s_intake_prepends_reprompt_to_last_assistant_turn(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        ctx.transcript = [
            Turn(
                role="user",
                text="hello",
                raw_text="hello",
                language="hi-IN",
            ),
            Turn(
                role="assistant",
                text="Aap kahaan rehte hain?",
                raw_text="Aap kahaan rehte hain?",
                language="hi-IN",
            ),
        ]
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, terminal = await mgr.handle_silence("test-call-001", 20.0)
        # Prefix + last question glued together
        assert "phir se poochta hoon" in spoken
        assert "Aap kahaan rehte hain?" in spoken
        assert terminal is False

    @pytest.mark.asyncio
    async def test_20s_intake_with_no_assistant_transcript_returns_bare_prefix(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        ctx.transcript = []  # no prior turns
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, terminal = await mgr.handle_silence("test-call-001", 20.0)
        # Prefix alone still returned
        assert "phir se poochta hoon" in spoken
        assert terminal is False

    @pytest.mark.asyncio
    async def test_20s_in_welcome_uses_short_english_language_reprompt(self):
        """During WELCOME (language-select) we must NOT replay the long
        multi-lingual greeting — we use a short English prompt instead."""
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.WELCOME)
        # Even though the welcome text is the "last assistant turn", the
        # phase-aware reprompt should IGNORE it and use the short prompt.
        ctx.transcript = [
            Turn(
                role="assistant",
                text="Namaste, Vanakkam, Hello. Vaidya here. Please say your language.",
                raw_text="Namaste, Vanakkam, Hello. Vaidya here. Please say your language.",
                language="hi-IN",
            ),
        ]
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, terminal = await mgr.handle_silence("test-call-001", 20.0)
        # Short English prompt, not the prefix + long welcome
        assert "Please say one language name" in spoken
        assert "Hindi, Tamil, Bengali" in spoken
        assert "Odia, or English" in spoken
        # The reprompt prefix ("Ek baar phir") should NOT be present
        assert "Ek baar phir" not in spoken
        assert "Namaste, Vanakkam" not in spoken
        assert terminal is False

    @pytest.mark.asyncio
    async def test_32s_returns_closure_terminal(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, terminal = await mgr.handle_silence("test-call-001", 32.0)
        assert "Line cut" in spoken or "Dhanyavaad" in spoken
        assert terminal is True
        assert ctx.phase == ConversationPhase.CLOSURE
        session.update.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_off_threshold_returns_empty_not_terminal(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, terminal = await mgr.handle_silence("test-call-001", 7.5)
        assert spoken == ""
        assert terminal is False

    @pytest.mark.asyncio
    async def test_respects_session_language(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="ta-IN", phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        spoken, _ = await mgr.handle_silence("test-call-001", 10.0)
        # Tamil nudge, not Hindi
        assert "ketkiren" in spoken.lower() or "sollunga" in spoken.lower()

    @pytest.mark.asyncio
    async def test_patient_silence_schedule_after_wait_request(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        ctx.metadata["silence_schedule"] = "patient"
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )

        early, terminal = await mgr.handle_silence("test-call-001", 10.0)
        assert early == ""
        assert terminal is False

        spoken, terminal = await mgr.handle_silence("test-call-001", 12.0)
        assert "yahin hoon" in spoken
        assert terminal is False


# ---------------------------------------------------------------------------
# switch_language (voice edge)
# ---------------------------------------------------------------------------


class TestSwitchLanguage:
    @pytest.mark.asyncio
    async def test_switches_to_supported_voice_language(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.switch_language("test-call-001", "ta-IN")
        assert result is True
        session.update.assert_awaited()
        # The updated context should carry the new language
        updated_ctx = session.update.await_args.args[0]
        assert updated_ctx.language == "ta-IN"
        assert updated_ctx.metadata["language_confirmed"] is True
        assert updated_ctx.metadata["language_source"] == "stt"

    @pytest.mark.asyncio
    async def test_same_language_confirms_choice_without_switching(self):
        """Speaking the default language is still a choice: no switch, but
        the welcome gate must release via ``language_confirmed``."""
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.WELCOME)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.switch_language("test-call-001", "hi-IN")
        assert result is False
        session.update.assert_awaited()
        updated_ctx = session.update.await_args.args[0]
        assert updated_ctx.language == "hi-IN"
        assert updated_ctx.metadata["language_confirmed"] is True
        assert updated_ctx.metadata["language_source"] == "stt"

    @pytest.mark.asyncio
    async def test_same_language_already_confirmed_is_a_noop(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        ctx.metadata["language_confirmed"] = True
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.switch_language("test-call-001", "hi-IN")
        assert result is False
        session.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_unsupported_voice_language(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(language="hi-IN", phase=ConversationPhase.INTAKE)
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.switch_language("test-call-001", "fr-FR")
        assert result is False
        session.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_switch_when_session_missing(self):
        orch, session, translator, audit, _ = _make_deps()
        session.get = AsyncMock(return_value=None)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        result = await mgr.switch_language("ghost-call", "ta-IN")
        assert result is False
        session.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# Dropped-call recovery (start_conversation resume policy)
# ---------------------------------------------------------------------------


class TestDroppedCallRecovery:
    @pytest.mark.asyncio
    async def test_welcome_phase_session_is_not_resumed(self):
        """A session still at the language gate has no progress worth
        resuming — the caller should get a fresh welcome instead."""
        orch, session, translator, audit, _ = _make_deps()
        stale = _make_context(call_id="old-call", phase=ConversationPhase.WELCOME)
        session.find_by_phone = AsyncMock(return_value="old-call")
        session.get = AsyncMock(return_value=stale)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        call_id, welcome = await mgr.start_conversation("hash123", "hi-IN", channel="voice")

        session.create.assert_awaited()  # fresh session, not a resume
        assert call_id != "old-call"
        assert "kat gaya" not in welcome

    @pytest.mark.asyncio
    async def test_intake_session_resumes_and_reasks_last_question(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(call_id="old-call", phase=ConversationPhase.INTAKE)
        ctx.add_turn(role="user", text="Bihar", raw_text="Bihar", language="hi-IN")
        ctx.add_turn(
            role="assistant",
            text="Aapke ghar mein kitne log hain?",
            raw_text="Aapke ghar mein kitne log hain?",
            language="hi-IN",
        )
        session.find_by_phone = AsyncMock(return_value="old-call")
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        call_id, welcome = await mgr.start_conversation("hash123", "hi-IN", channel="voice")

        assert call_id == "old-call"
        session.create.assert_not_awaited()
        # Resume notice followed by the pending question — no dead air.
        assert "cut ho gayi" in welcome
        assert welcome.endswith("Aapke ghar mein kitne log hain?")

    @pytest.mark.asyncio
    async def test_resume_without_transcript_still_returns_resume_notice(self):
        orch, session, translator, audit, _ = _make_deps()
        ctx = _make_context(call_id="old-call", phase=ConversationPhase.INTAKE)
        session.find_by_phone = AsyncMock(return_value="old-call")
        session.get = AsyncMock(return_value=ctx)

        mgr = ConversationManager(
            orchestrator=orch,
            session_manager=session,
            translator=translator,
            audit_trail=audit,
        )
        call_id, welcome = await mgr.start_conversation("hash123", "hi-IN", channel="voice")

        assert call_id == "old-call"
        assert "cut ho gayi" in welcome
