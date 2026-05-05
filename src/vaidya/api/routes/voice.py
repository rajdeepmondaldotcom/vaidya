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
import re
from html import escape

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

# Check whether the voice pipeline dependencies are available
try:
    from vaidya.telephony.pipeline import PIPECAT_AVAILABLE
except ImportError:
    PIPECAT_AVAILABLE = False

_PHONE_HASH_RE = re.compile(r"^[0-9a-f]{16,64}$")


def _hash_identifier(value: object) -> str:
    """Hash caller/call identifiers before they touch logs or session indexes."""
    raw = str(value or "").strip()
    if not raw:
        raw = "unknown"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_attr(value: object) -> str:
    """Escape values interpolated into TwiML attributes."""
    return escape(str(value or ""), quote=True)


def _phone_hash_from_call_data(call_data: dict) -> str:
    """Resolve the privacy-preserving session key from Twilio start data."""
    custom = call_data.get("body") or {}
    candidate = str(custom.get("phone_hash", "")).strip().lower()
    if candidate:
        if _PHONE_HASH_RE.fullmatch(candidate):
            return candidate
        return _hash_identifier(candidate)

    for key in ("call_id", "stream_id"):
        value = call_data.get(key)
        if value:
            return _hash_identifier(value)

    return _hash_identifier("twilio-unknown-call")


def _validate_twilio_http_request(
    request: Request,
    form,
    auth_token: str,
) -> bool:
    """Validate a Twilio webhook request using the official helper."""
    try:
        from twilio.request_validator import RequestValidator
    except ImportError:
        logger.error("Twilio auth token configured but twilio package is unavailable")
        return False

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        logger.warning("Missing Twilio signature")
        return False

    validator = RequestValidator(auth_token)
    return bool(validator.validate(str(request.url), form, signature))


def _validate_twilio_websocket(websocket: WebSocket, auth_token: str, public_url: str) -> bool:
    """Validate Twilio's WebSocket upgrade signature when configured."""
    try:
        from twilio.request_validator import RequestValidator
    except ImportError:
        logger.error("Twilio auth token configured but twilio package is unavailable")
        return False

    signature = websocket.headers.get("x-twilio-signature", "")
    if not signature:
        logger.warning("Missing Twilio WebSocket signature")
        return False

    url = public_url or str(websocket.url)
    validator = RequestValidator(auth_token)
    return bool(validator.validate(url, {}, signature))


def _build_stream_twiml(ws_url: str, phone_hash: str, status_callback_url: str = "") -> str:
    """Build TwiML for a bidirectional Twilio Media Stream."""
    attrs = [f'url="{_safe_attr(ws_url)}"']
    if status_callback_url:
        attrs.append(f'statusCallback="{_safe_attr(status_callback_url)}"')
        attrs.append('statusCallbackMethod="POST"')
    attr_text = " ".join(attrs)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream {attr_text}>
            <Parameter name="phone_hash" value="{_safe_attr(phone_hash)}" />
        </Stream>
    </Connect>
</Response>"""


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
    phone_hash = _hash_identifier(caller or call_sid)

    logger.info("Incoming call", extra={"caller_hash": phone_hash, "call_sid": call_sid})

    # Twilio signature verification
    if settings.twilio_auth_token and not _validate_twilio_http_request(
        request,
        form,
        settings.twilio_auth_token,
    ):
        return Response(status_code=403, content="Forbidden")

    if not ws_url:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>This service is not yet configured for phone calls. Please try again later.</Say>
    <Hangup/>
</Response>"""
        return Response(content=twiml, media_type="application/xml")

    twiml = _build_stream_twiml(
        ws_url=ws_url,
        phone_hash=phone_hash,
        status_callback_url=settings.voice_status_callback_url,
    )

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

    if settings.twilio_auth_token and not _validate_twilio_websocket(
        websocket,
        settings.twilio_auth_token,
        settings.voice_websocket_url,
    ):
        await websocket.close(code=1008, reason="invalid Twilio signature")
        return

    await websocket.accept()

    call_id: str | None = None
    try:
        from vaidya.telephony.pipeline import parse_voice_websocket_start

        transport_type, call_data = await parse_voice_websocket_start(websocket)
    except WebSocketDisconnect:
        logger.info("Caller disconnected before voice handshake")
        return
    except Exception as exc:
        logger.error("Voice handshake failed: %s", exc, exc_info=True)
        await websocket.close(code=1011, reason="voice handshake failed")
        return

    # Default language is used only for the short voice welcome + initial TTS
    # voice. The agent processor auto-detects the caller's actual language
    # from the first STT transcription and switches both session + TTS.
    language = settings.default_voice_language or "hi-IN"

    phone_hash = _phone_hash_from_call_data(call_data)

    mgr = websocket.app.state.conversation_manager
    call_id, welcome = await mgr.start_conversation(phone_hash, language, channel="voice")

    logger.info(
        "Voice session started",
        extra={"call_id": call_id, "transport_type": transport_type},
    )

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
            twilio_account_sid=settings.twilio_account_sid,
            twilio_auth_token=settings.twilio_auth_token,
            welcome_text=welcome,
            transport_type=transport_type,
            call_data=call_data,
        )
    except WebSocketDisconnect:
        logger.info("Caller disconnected", extra={"call_id": call_id})
    except Exception as exc:
        logger.error(
            "Voice pipeline error: %s: %s",
            type(exc).__name__,
            exc,
            extra={"call_id": call_id},
            exc_info=True,
        )
    finally:
        if call_id is not None:
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
