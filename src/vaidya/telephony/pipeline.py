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
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.runner.utils import parse_telephony_websocket
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.sarvam.stt import SarvamSTTService, SarvamSTTSettings
    from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

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

_TELEPHONY_SAMPLE_RATE = 8000


def _hash_identifier(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = "unknown"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def parse_voice_websocket_start(websocket: WebSocket) -> tuple[str, dict]:
    """Read and parse the telephony WebSocket start handshake."""
    if not PIPECAT_AVAILABLE:
        raise RuntimeError("pipecat-ai is required to parse telephony WebSocket handshakes")
    return await parse_telephony_websocket(websocket)


def _build_twilio_serializer(
    *,
    stream_sid: str,
    twilio_call_sid: str,
    twilio_account_sid: str,
    twilio_auth_token: str,
):
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


def _build_websocket_params(serializer):
    """Build FastAPI WebSocket transport params compatible with Pipecat 1.1."""
    return FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=_TELEPHONY_SAMPLE_RATE,
        audio_out_sample_rate=_TELEPHONY_SAMPLE_RATE,
        serializer=serializer,
        add_wav_header=False,
    )


def _build_stt_service(sarvam_api_key: str):
    """Build Sarvam STT with service-side VAD signals for telephony turn-taking."""
    return SarvamSTTService(
        api_key=sarvam_api_key,
        sample_rate=_TELEPHONY_SAMPLE_RATE,
        input_audio_codec="pcm_s16le",
        settings=SarvamSTTSettings(
            model="saaras:v3",
            language=None,
            vad_signals=True,
            high_vad_sensitivity=True,
        ),
    )


def _build_tts_service(sarvam_api_key: str, speaker: str, language: str):
    """Build Sarvam TTS for 8 kHz telephony output."""
    return SarvamTTSService(
        api_key=sarvam_api_key,
        sample_rate=_TELEPHONY_SAMPLE_RATE,
        settings=SarvamTTSSettings(model="bulbul:v3", voice=speaker, language=language),
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
    call_data: dict | None = None,
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

    from pipecat.frames.frames import TextFrame

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

    stt = _build_stt_service(sarvam_api_key)
    tts = _build_tts_service(sarvam_api_key, speaker, language)

    agent = VaidyaAgentProcessor(
        conversation_manager=conversation_manager,
        call_id=call_id,
        language=language,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            agent,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=_TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=_TELEPHONY_SAMPLE_RATE,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Voice client connected", extra={"call_id": call_id})
        if welcome_text:
            await task.queue_frames([TextFrame(text=welcome_text)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Voice client disconnected", extra={"call_id": call_id})

    runner = PipelineRunner()
    await runner.run(task)
