"""Twilio Media Streams frame serializer for Pipecat."""

from __future__ import annotations

import base64
import json
import logging

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import AudioRawFrame, Frame, OutputAudioRawFrame
    from pipecat.serializers.base_serializer import FrameSerializer

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

_PLACEHOLDER: type = type("_Placeholder", (), {})

_FrameSerializer = FrameSerializer if PIPECAT_AVAILABLE else _PLACEHOLDER


class TwilioFrameSerializer(_FrameSerializer):  # type: ignore[misc]
    """Translate between Twilio Media Streams WebSocket JSON and Pipecat frames."""

    def __init__(self) -> None:
        self._stream_sid: str = ""

    @property
    def stream_sid(self) -> str:
        return self._stream_sid

    def deserialize(self, data: bytes | str) -> Frame | None:
        try:
            msg = json.loads(data)
            event = msg.get("event", "")

            if event == "media":
                payload = msg["media"]["payload"]
                audio = base64.b64decode(payload)
                return AudioRawFrame(audio=audio, sample_rate=8000, num_channels=1)

            if event == "start":
                self._stream_sid = msg.get("streamSid", "")
                logger.info("Twilio stream started", extra={"stream_sid": self._stream_sid})

            return None
        except Exception:
            logger.debug("Failed to deserialize Twilio message", exc_info=True)
            return None

    def serialize(self, frame: Frame) -> bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            payload = base64.b64encode(frame.audio).decode("utf-8")
            msg = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }
            return json.dumps(msg).encode("utf-8")
        return None
