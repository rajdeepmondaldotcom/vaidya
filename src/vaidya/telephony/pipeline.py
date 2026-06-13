"""Voice pipeline: bridges Twilio telephony to Vaidya's conversation engine via Pipecat.

Architecture::

    Twilio WebSocket <-> TwilioFrameSerializer <-> Pipecat Transport
        -> STT (Sarvam Saaras v3)
        -> VaidyaAgentProcessor (our multi-agent orchestrator)
        -> TTS (Sarvam Bulbul v3)
        -> Pipecat Transport <-> TwilioFrameSerializer <-> Twilio WebSocket

Requires ``pipecat-ai[sarvam]>=1.0.0`` — guarded import so the rest of the app works without it.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, cast

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        Frame,
        InputAudioRawFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.runner.utils import parse_telephony_websocket
    from pipecat.services.sarvam.stt import SarvamSTTService, SarvamSTTSettings
    from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from vaidya.telephony.twilio_serializer import TwilioFrameSerializer

    PIPECAT_AVAILABLE = True
    logger.info("Pipecat voice pipeline imports OK — voice calls enabled")
except ImportError as _pipecat_import_err:
    PIPECAT_AVAILABLE = False
    logger.error(
        "Pipecat voice pipeline unavailable: %s",
        _pipecat_import_err,
    )

if TYPE_CHECKING:
    from fastapi import WebSocket

    from vaidya.pipeline.conversation import ConversationManager
    from vaidya.sarvam.cost import CostTracker

_TELEPHONY_SAMPLE_RATE = 8000
SarvamSTTMode = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]


def _coerce_stt_mode(mode: str) -> SarvamSTTMode:
    allowed: tuple[SarvamSTTMode, ...] = (
        "transcribe",
        "translate",
        "verbatim",
        "translit",
        "codemix",
    )
    if mode in allowed:
        return mode
    logger.warning("Invalid Sarvam STT mode %r; falling back to transcribe", mode)
    return "transcribe"


if PIPECAT_AVAILABLE:

    class GatedInterruptionSTTService(SarvamSTTService):
        """Sarvam STT that only interrupts when the bot is actually speaking.

        The stock service broadcasts a pipeline interruption on every VAD
        START_SPEECH — even on a quiet line. That interruption races the
        reply being generated for the very utterance that triggered it and
        clears its audio, so fast (canned) replies are never heard. True
        barge-in — the caller talking over the bot — still interrupts.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._vaidya_bot_speaking = False

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._vaidya_bot_speaking = True
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._vaidya_bot_speaking = False
            await super().process_frame(frame, direction)

        async def broadcast_interruption(self) -> None:
            if not self._vaidya_bot_speaking:
                logger.debug("Suppressing VAD interruption: bot is not speaking")
                return
            await super().broadcast_interruption()

    class SarvamStreamingSTTCostProcessor(FrameProcessor):
        """Counts streamed input audio seconds before Sarvam STT."""

        def __init__(
            self,
            *,
            costs: CostTracker,
            call_id: str,
            model: str,
            mode: str,
            with_diarization: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._costs = costs
            self._call_id = call_id
            self._model = model
            self._mode = mode
            self._with_diarization = with_diarization
            self._audio_seconds = 0.0
            self._recorded = False

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InputAudioRawFrame):
                sample_rate = getattr(frame, "sample_rate", 0) or 0
                if sample_rate > 0:
                    self._audio_seconds += frame.num_frames / sample_rate
            await self.push_frame(frame, direction)

        def record_final_cost(self) -> None:
            if self._recorded or self._audio_seconds <= 0:
                return
            self._recorded = True
            self._costs.record_stt(
                self._audio_seconds,
                call_id=self._call_id,
                model=self._model,
                mode=self._mode,
                with_diarization=self._with_diarization,
                metadata={
                    "api_mode": "pipecat_streaming",
                    "duration_source": "pipecat_input_audio_frames",
                    "sample_rate": _TELEPHONY_SAMPLE_RATE,
                },
            )

    class SarvamStreamingTTSCostProcessor(FrameProcessor):
        """Records text characters passed into Sarvam streaming TTS."""

        def __init__(
            self,
            *,
            costs: CostTracker,
            call_id: str,
            model: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._costs = costs
            self._call_id = call_id
            self._model = model

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
                text = getattr(frame, "text", "") or ""
                if text and not getattr(frame, "skip_tts", False):
                    self._costs.record_tts(
                        len(text),
                        call_id=self._call_id,
                        model=self._model,
                        mode="pipecat_streaming",
                        metadata={
                            "api_mode": "pipecat_streaming",
                            "text_frame": type(frame).__name__,
                        },
                    )
            await self.push_frame(frame, direction)


def _hash_identifier(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = "unknown"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def parse_voice_websocket_start(websocket: WebSocket) -> tuple[str, dict[str, Any]]:
    """Read and parse the telephony WebSocket start handshake."""
    if not PIPECAT_AVAILABLE:
        raise RuntimeError("pipecat-ai is required to parse telephony WebSocket handshakes")
    return cast(tuple[str, dict[str, Any]], await parse_telephony_websocket(websocket))


def _build_twilio_serializer(
    *,
    stream_sid: str,
    twilio_call_sid: str,
    twilio_account_sid: str,
    twilio_auth_token: str,
) -> Any:
    """Build the Twilio serializer with safe auto-hangup behavior."""
    auto_hang_up = bool(twilio_account_sid and twilio_auth_token and twilio_call_sid)
    return TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=twilio_call_sid or None,
        account_sid=twilio_account_sid or None,
        auth_token=twilio_auth_token or None,
        params=TwilioFrameSerializer.InputParams(
            auto_hang_up=auto_hang_up,
            sample_rate=_TELEPHONY_SAMPLE_RATE,
        ),
    )


def _build_websocket_params(serializer: Any) -> Any:
    """Build FastAPI WebSocket transport params compatible with Pipecat 1.1."""
    return FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=_TELEPHONY_SAMPLE_RATE,
        audio_out_sample_rate=_TELEPHONY_SAMPLE_RATE,
        serializer=serializer,
        add_wav_header=False,
    )


def _build_stt_service(
    sarvam_api_key: str,
    *,
    model: str = "saaras:v3",
    mode: str = "codemix",
    interrupt_min_speech_frames: int = 3,
) -> Any:
    """Build Sarvam STT with service-side VAD signals for telephony turn-taking."""
    # Must stay "wav": sarvamai's streaming AudioData only accepts
    # encoding="audio/wav" (raw PCM, rate from the connection params), and
    # Sarvam supports PCM codec labels at 16 kHz only — telephony is 8 kHz.
    return GatedInterruptionSTTService(
        api_key=sarvam_api_key,
        mode=_coerce_stt_mode(mode),
        sample_rate=_TELEPHONY_SAMPLE_RATE,
        input_audio_codec="wav",
        settings=SarvamSTTSettings(
            model=model,
            language=None,
            vad_signals=True,
            high_vad_sensitivity=True,
            interrupt_min_speech_frames=interrupt_min_speech_frames,
        ),
    )


def _build_tts_service(
    sarvam_api_key: str,
    speaker: str,
    language: str,
    *,
    model: str = "bulbul:v3",
    pace: float = 0.94,
    temperature: float = 0.55,
    min_buffer_size: int = 35,
    max_chunk_length: int = 130,
) -> Any:
    """Build Sarvam TTS for 8 kHz telephony output."""
    return SarvamTTSService(
        api_key=sarvam_api_key,
        sample_rate=_TELEPHONY_SAMPLE_RATE,
        settings=SarvamTTSSettings(
            model=model,
            voice=speaker,
            language=language,
            pace=pace,
            temperature=temperature,
            min_buffer_size=min_buffer_size,
            max_chunk_length=max_chunk_length,
        ),
    )


async def run_voice_pipeline(
    websocket: WebSocket,
    conversation_manager: ConversationManager,
    call_id: str,
    language: str,
    sarvam_api_key: str,
    speaker: str = "anushka",
    twilio_account_sid: str = "",
    twilio_auth_token: str = "",
    welcome_text: str = "",
    transport_type: str | None = None,
    call_data: dict[str, Any] | None = None,
    cost_tracker: CostTracker | None = None,
    stt_model: str = "saaras:v3",
    stt_mode: str = "codemix",
    stt_interrupt_min_speech_frames: int = 3,
    tts_model: str = "bulbul:v3",
    tts_pace: float = 0.94,
    tts_temperature: float = 0.55,
    tts_min_buffer_size: int = 35,
    tts_max_chunk_length: int = 130,
    playback_marks_enabled: bool = True,
    telephony_rate_inr_per_minute: float = 0.0,
    telephony_provider: str = "twilio",
) -> None:
    """Run a full voice call pipeline on the given WebSocket.

    The WebSocket must already be accepted. The route normally reads
    Twilio's initial handshake before session creation and passes it in;
    this function can still parse it as a fallback for direct callers.

    Args:
        websocket: The accepted FastAPI WebSocket (Twilio audio stream).
        conversation_manager: Vaidya's :class:`ConversationManager`.
        call_id: Unique session identifier for this call.
        language: BCP-47 language code (e.g. ``"hi-IN"``).
        sarvam_api_key: Sarvam AI API key for STT/TTS services.
        speaker: Sarvam TTS voice id (default ``"anushka"``).
        twilio_account_sid: Twilio account SID (required for auto hang-up).
        twilio_auth_token: Twilio auth token (required for auto hang-up).

    Raises:
        RuntimeError: If ``pipecat-ai`` is not installed.
    """
    if not PIPECAT_AVAILABLE:
        raise RuntimeError(
            "pipecat-ai[sarvam]>=1.0.0 is required for voice pipeline. "
            "Install with: pip install 'pipecat-ai[sarvam]>=1.0.0'"
        )

    from vaidya.telephony.agent_processor import VaidyaAgentProcessor

    # Read Twilio's "connected" + "start" messages only when the route has
    # not already consumed them to derive a privacy-preserving session key.
    if call_data is None:
        transport_type, call_data = await parse_voice_websocket_start(websocket)
    if transport_type != "twilio":
        raise RuntimeError(f"Unsupported voice transport: {transport_type or 'unknown'}")

    stream_sid = call_data.get("stream_id", "")
    twilio_call_sid = call_data.get("call_id", "")
    logger.info(
        "Twilio stream handshake received",
        extra={
            "call_id": call_id,
            "transport_type": transport_type,
            "stream_sid_hash": _hash_identifier(stream_sid) if stream_sid else "",
            "twilio_call_sid_hash": _hash_identifier(twilio_call_sid) if twilio_call_sid else "",
        },
    )

    serializer = _build_twilio_serializer(
        stream_sid=stream_sid,
        twilio_call_sid=twilio_call_sid,
        twilio_account_sid=twilio_account_sid,
        twilio_auth_token=twilio_auth_token,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=_build_websocket_params(serializer),
    )

    stt = _build_stt_service(
        sarvam_api_key,
        model=stt_model,
        mode=stt_mode,
        interrupt_min_speech_frames=stt_interrupt_min_speech_frames,
    )
    tts = _build_tts_service(
        sarvam_api_key,
        speaker,
        language,
        model=tts_model,
        pace=tts_pace,
        temperature=tts_temperature,
        min_buffer_size=tts_min_buffer_size,
        max_chunk_length=tts_max_chunk_length,
    )

    agent = VaidyaAgentProcessor(
        conversation_manager=conversation_manager,
        call_id=call_id,
        language=language,
        playback_marks_enabled=playback_marks_enabled,
    )

    stt_cost = (
        SarvamStreamingSTTCostProcessor(
            costs=cost_tracker,
            call_id=call_id,
            model=stt_model,
            mode=stt_mode,
        )
        if cost_tracker
        else None
    )
    tts_cost = (
        SarvamStreamingTTSCostProcessor(
            costs=cost_tracker,
            call_id=call_id,
            model=tts_model,
        )
        if cost_tracker
        else None
    )

    pipeline_steps: list[Any] = [transport.input()]
    if stt_cost is not None:
        pipeline_steps.append(stt_cost)
    pipeline_steps.extend([stt, agent])
    if tts_cost is not None:
        pipeline_steps.append(tts_cost)
    pipeline_steps.extend([tts, transport.output()])

    pipeline = Pipeline(pipeline_steps)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=_TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=_TELEPHONY_SAMPLE_RATE,
        ),
    )

    @transport.event_handler("on_client_connected")  # type: ignore[untyped-decorator]
    async def on_connected(transport: Any, client: Any) -> None:
        logger.info("Voice client connected", extra={"call_id": call_id})
        if welcome_text:
            # Envelope required: the TTS service only flushes its sentence
            # aggregation and Sarvam's server-side buffer on the end frame,
            # otherwise the welcome's last sentence is held until the next
            # turn's text arrives.
            await task.queue_frames(
                [
                    LLMFullResponseStartFrame(),
                    TextFrame(text=welcome_text),
                    LLMFullResponseEndFrame(),
                ]
            )

    @transport.event_handler("on_client_disconnected")  # type: ignore[untyped-decorator]
    async def on_disconnected(transport: Any, client: Any) -> None:
        logger.info("Voice client disconnected", extra={"call_id": call_id})

    runner = PipelineRunner()
    call_started_at = time.perf_counter()
    try:
        await runner.run(task)
    finally:
        if stt_cost is not None:
            stt_cost.record_final_cost()
        if cost_tracker is not None and telephony_rate_inr_per_minute > 0:
            cost_tracker.record_telephony(
                time.perf_counter() - call_started_at,
                rate_per_minute_inr=telephony_rate_inr_per_minute,
                call_id=call_id,
                provider=telephony_provider,
                metadata={"duration_source": "voice_pipeline_wall_time"},
            )
