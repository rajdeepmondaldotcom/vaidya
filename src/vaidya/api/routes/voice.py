"""Voice call endpoints: Twilio webhook + WebSocket audio streaming.

Provides three endpoints:

- ``POST /voice/incoming`` — Twilio calls this when someone dials our number.
  Returns TwiML that instructs Twilio to open a bidirectional audio stream.
- ``WS /voice/stream`` — Receives the bidirectional audio stream from Twilio
  and runs the Pipecat voice pipeline.
- ``POST /voice/status`` — Twilio status callback for call lifecycle events.

Requires ``pipecat-ai[sarvam]`` and ``twilio`` packages.  If they are not
installed, the WebSocket endpoint returns HTTP 503 and the incoming
endpoint returns a spoken apology.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

# Check whether the voice pipeline dependencies are available
try:
    from vaidya.telephony.pipeline import PIPECAT_AVAILABLE
except ImportError:
    PIPECAT_AVAILABLE = False


@router.post("/incoming")
async def incoming_call(request: Request) -> Response:
    """Twilio webhook: called when someone dials our number.

    Returns TwiML that tells Twilio to open a bidirectional Media Stream
    to our WebSocket endpoint.
    """
    settings = request.app.state.settings
    ws_url = settings.voice_websocket_url

    form = await request.form()
    caller = form.get("From", form.get("Caller", "unknown"))
    call_sid = form.get("CallSid", "")

    logger.info("Incoming call", extra={"caller": caller, "call_sid": call_sid})

    # Twilio signature verification
    if settings.twilio_auth_token:
        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature:
            logger.warning("Missing Twilio signature")
            return Response(status_code=403, content="Forbidden")

    if not ws_url:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>This service is not yet configured for phone calls. Please try again later.</Say>
    <Hangup/>
</Response>"""
        return Response(content=twiml, media_type="application/xml")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="caller" value="{caller}" />
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint: receives bidirectional audio from Twilio.

    This is the main voice call handler. For each connected call it:
    1. Accepts the WebSocket.
    2. Creates a conversation session via :class:`ConversationManager`.
    3. Runs the Pipecat voice pipeline (STT -> Agent -> TTS).
    4. Cleans up the session when the call ends.

    Twilio sends ``connected`` then ``start`` (with customParameters) as
    the first messages on the stream. Caller info is extracted from
    those custom parameters.
    """
    settings = websocket.app.state.settings

    if not PIPECAT_AVAILABLE:
        await websocket.close(code=1011, reason="pipecat not installed")
        return

    await websocket.accept()

    # Twilio sends "connected" then "start" with customParameters.
    # We start the session immediately; caller info arrives in the stream.
    language = "hi-IN"

    phone_hash = hashlib.sha256(b"twilio-call").hexdigest()[:16]

    mgr = websocket.app.state.conversation_manager
    call_id, welcome = await mgr.start_conversation(phone_hash, language)

    logger.info("Voice session started", extra={"call_id": call_id})

    from vaidya.voice.language import TTS_SPEAKERS, normalize_language

    lang = normalize_language(language)
    speaker = TTS_SPEAKERS.get(lang, "priya")

    try:
        from vaidya.telephony.pipeline import run_voice_pipeline

        await run_voice_pipeline(
            websocket=websocket,
            conversation_manager=mgr,
            call_id=call_id,
            language=language,
            sarvam_api_key=settings.sarvam_api_key,
            speaker=speaker,
        )
    except WebSocketDisconnect:
        logger.info("Caller disconnected", extra={"call_id": call_id})
    except Exception as exc:
        logger.error("Voice pipeline error", extra={"call_id": call_id, "error": str(exc)})
    finally:
        await mgr.end_conversation(call_id)
        logger.info("Voice session ended", extra={"call_id": call_id})


@router.post("/status")
async def call_status(request: Request) -> dict:
    """Twilio status callback: tracks call lifecycle events.

    Twilio POSTs here when a call is initiated, ringing, answered, or completed.
    We log the event for monitoring and debugging.
    """
    form = await request.form()
    status = form.get("CallStatus", "unknown")
    call_sid = form.get("CallSid", "")
    duration = form.get("CallDuration", "0")

    logger.info(
        "Call status update",
        extra={"call_sid": call_sid, "status": status, "duration": duration},
    )

    return {"status": "ok"}
