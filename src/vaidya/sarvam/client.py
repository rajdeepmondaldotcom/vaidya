"""Async wrapper around sarvamai.SarvamAI for non-blocking FastAPI integration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from sarvamai import SarvamAI

logger = logging.getLogger(__name__)


class SarvamClient:
    """Thin async facade over the synchronous sarvamai SDK."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = SarvamAI(api_subscription_key=api_key)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        wiki_grounding: bool = False,
    ) -> str:
        """Call Sarvam LLM and return the response text."""
        # Use asyncio.to_thread to avoid blocking the event loop
        # sarvamai SDK is synchronous
        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            if wiki_grounding:
                kwargs["wiki_grounding"] = wiki_grounding
            response = await asyncio.to_thread(
                self._client.chat.completions,
                **kwargs,
            )
            elapsed = (time.perf_counter() - start) * 1000
            content = response.choices[0].message.content
            logger.info("LLM call", extra={"model": model, "latency_ms": f"{elapsed:.0f}"})
            return content
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "LLM call failed",
                extra={"model": model, "error": str(e), "latency_ms": f"{elapsed:.0f}"},
            )
            raise

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        wiki_grounding: bool = False,
    ) -> dict[str, Any]:
        """Call LLM and parse JSON response, with code-fence stripping."""
        raw = await self.chat(
            model,
            messages,
            temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            wiki_grounding=wiki_grounding,
        )
        return parse_llm_json(raw)

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str = "mayura:v1",
        mode: str = "modern-colloquial",
        speaker_gender: str = "Male",
        output_script: str = "fully-native",
        numerals_format: str = "international",
    ) -> str:
        """Translate text using Mayura (colloquial) or Sarvam Translate (formal).

        Docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models/mayura
        """
        if source_lang == target_lang:
            return text
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self._client.text.translate,
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang,
                mode=mode,
                model=model,
                speaker_gender=speaker_gender,
                output_script=output_script,
                numerals_format=numerals_format,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "Translation",
                extra={
                    "model": model,
                    "src": source_lang,
                    "tgt": target_lang,
                    "latency_ms": f"{elapsed:.0f}",
                },
            )
            return response.translated_text
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Translation failed",
                extra={"model": model, "error": str(e), "latency_ms": f"{elapsed:.0f}"},
            )
            raise

    async def tts(
        self,
        text: str,
        language: str,
        speaker: str = "anushka",
        model: str = "bulbul:v3",
        temperature: float = 0.6,
        pace: float = 1.0,
        speech_sample_rate: int = 8000,
        output_audio_codec: str = "wav",
    ) -> bytes | None:
        """Convert text to speech using Bulbul v3.

        Docs: https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert
        Supports 45 speakers, 11 languages, temperature control (v3 only).
        bulbul:v3 supports up to 2500 chars (v2 was 1500).
        """
        truncated = text[:2500]
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self._client.text_to_speech.convert,
                text=truncated,
                target_language_code=language,
                speaker=speaker,
                model=model,
                enable_preprocessing=True,
                pace=pace,
                speech_sample_rate=speech_sample_rate,
                output_audio_codec=output_audio_codec,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "TTS",
                extra={
                    "model": model,
                    "speaker": speaker,
                    "lang": language,
                    "latency_ms": f"{elapsed:.0f}",
                },
            )
            return response.audios[0] if response.audios else None
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "TTS failed",
                extra={"model": model, "error": str(e), "latency_ms": f"{elapsed:.0f}"},
            )
            return None

    async def stt(
        self,
        audio_file: Any,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        language: str | None = None,
    ) -> tuple[str, str, float]:
        """Transcribe audio using Saaras v3 (23 Indian languages).

        Returns (transcript, language_code, language_probability).

        Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
        Modes: transcribe, translate, verbatim, translit, codemix
        """
        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "file": audio_file,
                "model": model,
                "mode": mode,
            }
            if language:
                kwargs["language_code"] = language
            response = await asyncio.to_thread(
                self._client.speech_to_text.transcribe,
                **kwargs,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "STT",
                extra={"model": model, "mode": mode, "latency_ms": f"{elapsed:.0f}"},
            )
            return (
                response.transcript,
                response.language_code,
                response.language_probability,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "STT failed",
                extra={"model": model, "error": str(e), "latency_ms": f"{elapsed:.0f}"},
            )
            raise

    async def stream_stt(
        self,
        audio_data: bytes,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        language: str | None = None,
        sample_rate: int = 8000,
    ) -> Any:
        """WebSocket streaming STT using AsyncSarvamAI."""
        from sarvamai import AsyncSarvamAI

        async_client = AsyncSarvamAI(api_subscription_key=self._api_key)
        kwargs: dict[str, Any] = {
            "model": model,
            "mode": mode,
            "sample_rate": sample_rate,
            "high_vad_sensitivity": True,
            "vad_signals": True,
        }
        if language:
            kwargs["language_code"] = language
        async with async_client.speech_to_text_streaming.connect(**kwargs) as ws:
            await ws.transcribe(audio=audio_data)
            response = await ws.recv()
            return response

    async def stream_tts(
        self,
        text: str,
        language: str,
        speaker: str = "anushka",
        model: str = "bulbul:v3",
    ) -> Any:
        """WebSocket streaming TTS using AsyncSarvamAI."""
        from sarvamai import AsyncSarvamAI

        async_client = AsyncSarvamAI(api_subscription_key=self._api_key)
        async with async_client.text_to_speech_streaming.connect(
            model=model,
            speaker=speaker,
            target_language_code=language,
        ) as ws:
            await ws.send(text=text)
            audio = await ws.recv()
            return audio

    # ------------------------------------------------------------------
    # Document Intelligence (Sarvam Vision) — ₹1.5/page
    # SDK: client.document_intelligence (job-based: initialise→upload→start→poll→download)
    # Phase 2: full implementation for WhatsApp document verification
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Language Identification — ₹3.5/10K chars
    # ------------------------------------------------------------------

    async def identify_language(self, text: str) -> tuple[str, str]:
        """Identify the language and script of input text.

        Returns (language_code, script_code) e.g. ("hi-IN", "Deva").
        Uses client.text.identify_language() per sarvamai SDK.
        """
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self._client.text.identify_language,
                input=text,
            )
            elapsed = (time.perf_counter() - start) * 1000
            lang = response.language_code
            script = getattr(response, "script_code", "")
            logger.info(
                "Language identification",
                extra={
                    "detected": lang,
                    "script": script,
                    "latency_ms": f"{elapsed:.0f}",
                },
            )
            return lang, script
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Language detection failed",
                extra={"error": str(e), "latency_ms": f"{elapsed:.0f}"},
            )
            raise

    # ------------------------------------------------------------------
    # Transliteration — ₹20/10K chars
    # ------------------------------------------------------------------

    async def transliterate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Transliterate text between scripts (e.g. Devanagari to Roman).

        Useful for logging romanized versions of Indic-script text.
        """
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                self._client.text.transliterate,
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "Transliteration",
                extra={"src": source_lang, "tgt": target_lang, "latency_ms": f"{elapsed:.0f}"},
            )
            return response.transliterated_text
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Transliteration failed",
                extra={"error": str(e), "latency_ms": f"{elapsed:.0f}"},
            )
            raise


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown code fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (code fence markers)
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse LLM JSON", extra={"raw": raw[:200]})
        return {"_raw": cleaned, "_parse_error": True}
