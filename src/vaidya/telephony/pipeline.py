"""Voice pipeline: bridges Twilio telephony to Vaidya's conversation engine via Pipecat.

Architecture::

    Twilio WebSocket <-> TwilioFrameSerializer <-> Pipecat Transport
        -> STT (Sarvam Saaras v3)
        -> VaidyaAgentProcessor (our multi-agent orchestrator)
        -> TTS (Sarvam Bulbul v3)
        -> Pipecat Transport <-> TwilioFrameSerializer <-> Twilio WebSocket

Requires ``pipecat-ai[sarvam]`` — guarded import so the rest of the app works without it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.services.sarvam.stt import SarvamSTTService
    from pipecat.services.sarvam.tts import SarvamTTSService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

if TYPE_CHECKING:
    from fastapi import WebSocket

    from vaidya.pipeline.conversation import ConversationManager


async def run_voice_pipeline(
    websocket: WebSocket,
    conversation_manager: ConversationManager,
    call_id: str,
    language: str,
    sarvam_api_key: str,
    speaker: str = "priya",
) -> None:
    """Run a full voice call pipeline on the given WebSocket.

    This is the main entry point for a single phone call. It blocks until
    the caller hangs up or the WebSocket disconnects.

    Args:
        websocket: The accepted FastAPI WebSocket (Twilio audio stream).
        conversation_manager: Vaidya's :class:`ConversationManager`.
        call_id: Unique session identifier for this call.
        language: BCP-47 language code (e.g. ``"hi-IN"``).
        sarvam_api_key: Sarvam AI API key for STT/TTS services.
        speaker: TTS speaker name (default ``"priya"``).

    Raises:
        RuntimeError: If ``pipecat-ai`` is not installed.
    """
    if not PIPECAT_AVAILABLE:
        raise RuntimeError(
            "pipecat-ai[sarvam] is required for voice pipeline. "
            "Install with: pip install 'pipecat-ai[sarvam]'"
        )

    from vaidya.telephony.agent_processor import VaidyaAgentProcessor
    from vaidya.telephony.twilio_serializer import TwilioFrameSerializer

    serializer = TwilioFrameSerializer()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
            add_wav_header=False,
        ),
    )

    stt = SarvamSTTService(
        api_key=sarvam_api_key,
        language=language,
        model="saaras:v3",
    )

    tts = SarvamTTSService(
        api_key=sarvam_api_key,
        target_language_code=language,
        model="bulbul:v3",
        speaker=speaker,
        speech_sample_rate=8000,
    )

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

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Voice client connected", extra={"call_id": call_id})

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Voice client disconnected", extra={"call_id": call_id})

    runner = PipelineRunner()
    await runner.run(task)
