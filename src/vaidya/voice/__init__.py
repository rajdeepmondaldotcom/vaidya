"""Voice layer: language detection, STT, and TTS."""

from vaidya.voice.language import Language, is_supported, normalize_language
from vaidya.voice.stt import STTClient
from vaidya.voice.tts import TTSClient

__all__ = [
    "Language",
    "STTClient",
    "TTSClient",
    "is_supported",
    "normalize_language",
]
