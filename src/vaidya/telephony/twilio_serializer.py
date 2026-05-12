"""Twilio Media Streams serializer extensions for playback marks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import Frame
    from pipecat.serializers.twilio import TwilioFrameSerializer as _PipecatTwilioFrameSerializer

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

    @dataclass
    class Frame:  # type: ignore[no-redef]
        pass

    class _PipecatTwilioFrameSerializer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def serialize(self, frame: Frame) -> str | bytes | None:
            del frame
            return None

        async def deserialize(self, data: str | bytes) -> Frame | None:
            del data
            return None


@dataclass
class TwilioPlaybackMarkRequestFrame(Frame):
    """Outbound control frame asking Twilio to ack playback completion."""

    mark_name: str


@dataclass
class TwilioPlaybackMarkFrame(Frame):
    """Inbound Twilio mark event returned after buffered audio has played."""

    mark_name: str
    stream_sid: str = ""


class TwilioFrameSerializer(_PipecatTwilioFrameSerializer):
    """Pipecat Twilio serializer with playback-mark request/ack support."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, TwilioPlaybackMarkRequestFrame):
            stream_sid = getattr(self, "_stream_sid", "")
            return json.dumps(
                {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": frame.mark_name},
                }
            )
        return await super().serialize(frame)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
        except (TypeError, json.JSONDecodeError):
            logger.debug("Failed to decode Twilio websocket message", exc_info=True)
            return await super().deserialize(data)

        if message.get("event") == "mark":
            mark = message.get("mark", {})
            return TwilioPlaybackMarkFrame(
                mark_name=str(mark.get("name", "")),
                stream_sid=str(message.get("streamSid", "")),
            )

        return await super().deserialize(data)
