"""Async wrapper around sarvamai.SarvamAI for non-blocking FastAPI integration."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time
import wave
from typing import Any, cast

from sarvamai import SarvamAI

from vaidya.sarvam.models import TTS_MAX_CHARS_V3
from vaidya.sarvam.resilience import CircuitOpenError, ServiceCircuitBreakers

logger = logging.getLogger(__name__)

# Trailing commas before } or ] — frequent LLM JSON malformation.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

# Free/credit-tier hard cap on completion tokens (a higher value 400s).
_MAX_OUTPUT_TOKENS = 4096

_VALID_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


def _coerce_reasoning_effort(value: str | None) -> str:
    """Map any input to a server-accepted reasoning_effort (low/medium/high).

    The API rejects anything else, and OMITTING the field triggers a
    verbose default that starves the content channel — so we never return
    None. Unknown / "none" / empty all fall back to the fast "low" floor.
    """
    if value and value.lower() in _VALID_REASONING_EFFORTS:
        return value.lower()
    return "low"


async def _retry_async(
    fn: Any,
    *args: Any,
    retries: int = 2,
    base_delay: float = 0.5,
    timeout: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Execute *fn* in a thread with timeout and exponential-backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn, *args, **kwargs),
                timeout=timeout,
            )
        except TimeoutError:
            last_exc = TimeoutError(f"API call timed out after {timeout}s")
            if attempt < retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "API call timed out, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    retries,
                )
                await asyncio.sleep(delay)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "API call failed, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    retries,
                    extra={"error": str(exc)},
                )
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _extract_chat_content(response: Any) -> str:
    """Extract text content from an LLM chat response.

    Handles the standard ``response.choices[0].message.content`` path and
    falls back to ``reasoning_content`` when the model returns reasoning
    output only.
    """
    msg = response.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)
    if not content and reasoning:
        logger.info("LLM returned reasoning_content, using as content")
        content = reasoning
    return content


def _model_latency_class(model: str) -> str:
    """Return Vaidya's routing class for Sarvam chat models."""
    if model == "sarvam-30b":
        return "fast"
    if model == "sarvam-105b":
        return "regular"
    return "unknown"


def _duration_from_wav_bytes(data: bytes) -> float | None:
    """Return WAV duration in seconds when *data* is a readable WAV payload."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return float(wav_file.getnframes() / frame_rate)
    except (EOFError, wave.Error):
        return None


def _duration_from_wav_path(path: str | os.PathLike[str]) -> float | None:
    """Return WAV duration in seconds when *path* points to a readable WAV file."""
    try:
        with wave.open(os.fspath(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return float(wav_file.getnframes() / frame_rate)
    except (EOFError, OSError, wave.Error):
        return None


def _read_file_like_bytes(audio_file: Any) -> bytes | None:
    """Read a seekable file-like object without changing its current position."""
    if isinstance(audio_file, io.BytesIO):
        return audio_file.getvalue()

    tell = getattr(audio_file, "tell", None)
    seek = getattr(audio_file, "seek", None)
    read = getattr(audio_file, "read", None)
    if not callable(tell) or not callable(seek) or not callable(read):
        return None

    try:
        position = tell()
        data = read()
        seek(position)
    except (OSError, ValueError):
        return None
    return bytes(data) if isinstance(data, (bytes, bytearray)) else None


def _estimate_audio_duration_seconds(
    audio_file: Any,
    *,
    raw_sample_rate: int | None = None,
    raw_sample_width_bytes: int = 2,
    raw_channels: int = 1,
) -> tuple[float | None, str]:
    """Best-effort audio duration for billing, without consuming file handles."""
    if isinstance(audio_file, (str, os.PathLike)):
        duration = _duration_from_wav_path(audio_file)
        return (duration, "wav_path") if duration is not None else (None, "unknown")

    data: bytes | None
    if isinstance(audio_file, (bytes, bytearray)):
        data = bytes(audio_file)
    else:
        data = _read_file_like_bytes(audio_file)

    if not data:
        return None, "unknown"

    wav_duration = _duration_from_wav_bytes(data)
    if wav_duration is not None:
        return wav_duration, "wav_bytes"

    if raw_sample_rate and raw_sample_rate > 0 and raw_sample_width_bytes > 0 and raw_channels > 0:
        return (
            len(data) / (raw_sample_rate * raw_sample_width_bytes * raw_channels),
            "raw_pcm_bytes",
        )

    return None, "unknown"


def _duration_from_stt_response(response: Any) -> float | None:
    """Extract audio duration from STT timestamps when the API response includes them."""
    timestamps = getattr(response, "timestamps", None)
    candidates: list[Any] = []
    if timestamps:
        if isinstance(timestamps, dict):
            candidates.append(timestamps.get("end_time_seconds"))
            nested = timestamps.get("timestamps")
            if isinstance(nested, dict):
                candidates.append(nested.get("end_time_seconds"))
        else:
            candidates.append(getattr(timestamps, "end_time_seconds", None))

    diarized = getattr(response, "diarized_transcript", None)
    entries = None
    if isinstance(diarized, dict):
        entries = diarized.get("entries")
    elif diarized is not None:
        entries = getattr(diarized, "entries", None)
    if entries:
        candidates.append(
            [
                entry.get("end_time_seconds")
                if isinstance(entry, dict)
                else getattr(entry, "end_time_seconds", None)
                for entry in entries
            ]
        )

    end_times: list[float] = []
    for candidate in candidates:
        values = candidate if isinstance(candidate, (list, tuple)) else [candidate]
        for value in values:
            if isinstance(value, (int, float)):
                end_times.append(float(value))

    return max(end_times) if end_times else None


class SarvamClient:
    """Thin async facade over the synchronous sarvamai SDK with cost tracking."""

    def __init__(
        self,
        api_key: str,
        retry_max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._client = SarvamAI(api_subscription_key=api_key)
        self._active_call_id: str = ""
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay = retry_base_delay
        self._timeout = timeout
        self._circuit_breakers = ServiceCircuitBreakers()

        from vaidya.sarvam.cost import CostTracker

        self.costs = CostTracker()

    @property
    def _retries(self) -> int:
        """Number of retries (attempts minus one)."""
        return max(0, self._retry_max_attempts - 1)

    def set_active_call_id(self, call_id: str) -> None:
        """Set the call_id used for cost attribution on subsequent API calls."""
        self._active_call_id = call_id

    def clear_active_call_id(self) -> None:
        """Clear the active call_id after a turn completes."""
        self._active_call_id = ""

    _SERVICE_BREAKER_KEYS: dict[str, str] = {
        "LLM call": "llm",
        "Translation": "translate",
        "TTS": "tts",
        "STT": "stt",
        "Language identification": "language_id",
        "Transliteration": "transliterate",
    }

    async def _timed_api_call(
        self,
        service_name: str,
        api_call: Any,
        cost_recorder: Any,
        result_extractor: Any,
        log_extras: dict[str, Any],
        *,
        on_error: str = "raise",
    ) -> Any:
        """Execute an API call with timing, cost tracking, and circuit breaker."""
        breaker_key = self._SERVICE_BREAKER_KEYS.get(service_name, service_name)
        cb = self._circuit_breakers.get(breaker_key)
        try:
            cb.check()
        except CircuitOpenError:
            logger.warning("Circuit open for %s, skipping call", service_name)
            if on_error == "return_none":
                return None
            raise

        start = time.perf_counter()
        try:
            response = await api_call()
            elapsed = (time.perf_counter() - start) * 1000
            result = result_extractor(response)
            cost_recorder(elapsed, response)
            logger.info(service_name, extra={**log_extras, "latency_ms": f"{elapsed:.0f}"})
            cb.record_success()
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "%s failed",
                service_name,
                extra={"error": str(e), "latency_ms": f"{elapsed:.0f}", **log_extras},
            )
            cb.record_failure()
            if on_error == "return_none":
                return None
            raise

    @staticmethod
    def _build_chat_kwargs(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        *,
        reasoning_effort: str | None = None,
        wiki_grounding: bool = False,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Assemble keyword arguments for the chat completions SDK call."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # The API hard-caps max_tokens at 4096 (free/credit tier); a
            # higher value is a 400. Reasoning models also need a generous
            # budget or content truncates to empty, so clamp UP into a safe
            # band rather than letting tiny values starve the output.
            "max_tokens": max(1024, min(int(max_tokens), _MAX_OUTPUT_TOKENS)),
        }
        optional = {
            # The API ONLY accepts low/medium/high. Omitting it makes the
            # model reason at a verbose default (all budget -> reasoning,
            # content empty), so always coerce to a valid value; "low" is
            # the fast, reliable floor.
            "reasoning_effort": _coerce_reasoning_effort(reasoning_effort),
            "wiki_grounding": wiki_grounding or None,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "tools": tools,
        }
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        return kwargs

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        wiki_grounding: bool = False,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Call Sarvam LLM and return the response text."""
        kwargs = self._build_chat_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            reasoning_effort=reasoning_effort,
            wiki_grounding=wiki_grounding,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            seed=seed,
            tools=tools,
        )

        async def api_call() -> Any:
            return await _retry_async(
                self._client.chat.completions,
                retries=self._retries,
                base_delay=self._retry_base_delay,
                timeout=self._timeout,
                **kwargs,
            )

        def record_cost(elapsed_ms: float, response: Any) -> None:
            tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
            self.costs.record_llm(
                tokens,
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                model=model,
                mode=reasoning_effort or "default",
                metadata={
                    "latency_class": _model_latency_class(model),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "wiki_grounding": wiki_grounding,
                },
            )

        return cast(
            str,
            await self._timed_api_call(
                "LLM call",
                api_call,
                record_cost,
                _extract_chat_content,
                {"model": model},
            ),
        )

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
        if not reasoning_effort or reasoning_effort == "none":
            reasoning_effort = None  # also skips the pointless no-reasoning retry
        raw = await self.chat(
            model,
            messages,
            temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            wiki_grounding=wiki_grounding,
        )
        parsed = parse_llm_json(raw)
        if parsed.get("_parse_error"):
            # Retry once at minimal explicit reasoning. Never omit the
            # param: the model then reasons at a verbose default and the
            # JSON often never reaches the content channel at all.
            logger.info("Retrying JSON chat with reasoning_effort=low")
            raw = await self.chat(
                model,
                messages,
                temperature,
                max_tokens=max_tokens,
                reasoning_effort="low",
                wiki_grounding=wiki_grounding,
            )
            parsed = parse_llm_json(raw)
        return parsed

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
        """Translate text between languages using Mayura or Sarvam Translate."""
        if source_lang == target_lang:
            return text

        async def api_call() -> Any:
            return await _retry_async(
                self._client.text.translate,
                retries=self._retries,
                base_delay=self._retry_base_delay,
                timeout=self._timeout,
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang,
                mode=mode,
                model=model,
                speaker_gender=speaker_gender,
                output_script=output_script,
                numerals_format=numerals_format,
            )

        def record_cost(elapsed_ms: float, _response: Any) -> None:
            self.costs.record_translate(
                len(text),
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                model=model,
                mode=mode,
                metadata={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "speaker_gender": speaker_gender,
                    "output_script": output_script,
                    "numerals_format": numerals_format,
                },
            )

        return cast(
            str,
            await self._timed_api_call(
                "Translation",
                api_call,
                record_cost,
                lambda r: r.translated_text,
                {"model": model, "src": source_lang, "tgt": target_lang, "chars": len(text)},
            ),
        )

    async def tts(
        self,
        text: str,
        language: str,
        speaker: str = "priya",
        model: str = "bulbul:v3",
        temperature: float = 0.6,
        pace: float = 1.0,
        speech_sample_rate: int = 8000,
        output_audio_codec: str = "wav",
    ) -> bytes | None:
        """Convert text to speech using Bulbul v3."""
        truncated = text[:TTS_MAX_CHARS_V3]

        async def api_call() -> Any:
            return await _retry_async(
                self._client.text_to_speech.convert,
                retries=self._retries,
                base_delay=self._retry_base_delay,
                timeout=self._timeout,
                text=truncated,
                target_language_code=language,
                speaker=speaker,
                model=model,
                temperature=temperature,
                pace=pace,
                speech_sample_rate=speech_sample_rate,
                output_audio_codec=output_audio_codec,
            )

        def record_cost(elapsed_ms: float, _response: Any) -> None:
            self.costs.record_tts(
                len(truncated),
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                model=model,
                mode="rest",
                metadata={
                    "language": language,
                    "speaker": speaker,
                    "pace": pace,
                    "temperature": temperature,
                    "speech_sample_rate": speech_sample_rate,
                    "output_audio_codec": output_audio_codec,
                    "truncated": len(truncated) < len(text),
                },
            )

        return cast(
            bytes | None,
            await self._timed_api_call(
                "TTS",
                api_call,
                record_cost,
                lambda r: r.audios[0] if r.audios else None,
                {"model": model, "speaker": speaker, "lang": language},
                on_error="return_none",
            ),
        )

    @staticmethod
    def _build_stt_kwargs(
        audio_file: Any,
        model: str,
        mode: str,
        *,
        language: str | None = None,
        with_timestamps: bool = False,
        with_diarization: bool = False,
        num_speakers: int | None = None,
    ) -> dict[str, Any]:
        """Assemble keyword arguments for the STT transcribe SDK call."""
        kwargs: dict[str, Any] = {"file": audio_file, "model": model, "mode": mode}
        optional = {
            "language_code": language,
            "with_timestamps": with_timestamps or None,
            "with_diarization": with_diarization or None,
            "num_speakers": num_speakers,
        }
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        return kwargs

    async def stt(
        self,
        audio_file: Any,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        language: str | None = None,
        with_timestamps: bool = False,
        with_diarization: bool = False,
        num_speakers: int | None = None,
    ) -> tuple[str, str, float]:
        """Transcribe audio and return (transcript, language_code, probability)."""
        kwargs = self._build_stt_kwargs(
            audio_file,
            model,
            mode,
            language=language,
            with_timestamps=with_timestamps,
            with_diarization=with_diarization,
            num_speakers=num_speakers,
        )

        async def api_call() -> Any:
            return await _retry_async(
                self._client.speech_to_text.transcribe,
                retries=self._retries,
                base_delay=self._retry_base_delay,
                timeout=self._timeout,
                **kwargs,
            )

        estimated_duration, estimated_source = _estimate_audio_duration_seconds(audio_file)

        def record_cost(elapsed_ms: float, response: Any) -> None:
            response_duration = _duration_from_stt_response(response)
            if response_duration is not None:
                duration_seconds = response_duration
                duration_source = "response_timestamps"
            elif estimated_duration is not None:
                duration_seconds = estimated_duration
                duration_source = estimated_source
            else:
                duration_seconds = elapsed_ms / 1000.0
                duration_source = "api_latency_fallback"
            self.costs.record_stt(
                duration_seconds,
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                model=model,
                mode=mode,
                with_diarization=with_diarization,
                metadata={
                    "api_mode": "rest",
                    "duration_source": duration_source,
                    "language": language,
                    "with_timestamps": with_timestamps,
                    "num_speakers": num_speakers,
                },
            )

        def extract(response: Any) -> tuple[str, str, float]:
            return (response.transcript, response.language_code, response.language_probability)

        return cast(
            tuple[str, str, float],
            await self._timed_api_call(
                "STT",
                api_call,
                record_cost,
                extract,
                {"model": model, "mode": mode},
            ),
        )

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
        start = time.perf_counter()
        async with async_client.speech_to_text_streaming.connect(**kwargs) as ws:
            await ws.transcribe(audio=audio_data)
            response = await ws.recv()
            elapsed_ms = (time.perf_counter() - start) * 1000
            duration_seconds, duration_source = _estimate_audio_duration_seconds(
                audio_data,
                raw_sample_rate=sample_rate,
            )
            self.costs.record_stt(
                duration_seconds if duration_seconds is not None else elapsed_ms / 1000.0,
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                model=model,
                mode=mode,
                metadata={
                    "api_mode": "streaming",
                    "duration_source": duration_source
                    if duration_seconds is not None
                    else "api_latency_fallback",
                    "language": language,
                    "sample_rate": sample_rate,
                },
            )
            return response

    async def stream_tts(
        self,
        text: str,
        language: str,
        speaker: str = "priya",
        model: str = "bulbul:v3",
    ) -> Any:
        """WebSocket streaming TTS using AsyncSarvamAI."""
        from sarvamai import AsyncSarvamAI

        async_client = AsyncSarvamAI(api_subscription_key=self._api_key)
        start = time.perf_counter()
        async with async_client.text_to_speech_streaming.connect(
            model=model,
            speaker=speaker,
            target_language_code=language,
        ) as ws:
            await ws.send(text=text)
            audio = await ws.recv()
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.costs.record_tts(
                len(text),
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                model=model,
                mode="streaming",
                metadata={"language": language, "speaker": speaker},
            )
            return audio

    async def identify_language(self, text: str) -> tuple[str, str]:
        """Identify the language and script of input text.

        Returns (language_code, script_code) e.g. ("hi-IN", "Deva").
        """

        async def api_call() -> Any:
            return await _retry_async(
                self._client.text.identify_language,
                retries=self._retries,
                base_delay=self._retry_base_delay,
                timeout=self._timeout,
                input=text,
            )

        def record_cost(elapsed_ms: float, _response: Any) -> None:
            self.costs.record_language_id(
                len(text),
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                metadata={"api_mode": "text"},
            )

        def extract(response: Any) -> tuple[str, str]:
            return response.language_code, getattr(response, "script_code", "")

        return cast(
            tuple[str, str],
            await self._timed_api_call(
                "Language identification",
                api_call,
                record_cost,
                extract,
                {},
            ),
        )

    async def transliterate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        numerals_format: str = "international",
        spoken_form: bool = False,
        spoken_form_numerals_language: str = "english",
    ) -> str:
        """Transliterate text between scripts (e.g. Devanagari to Roman)."""
        tl_kwargs: dict[str, Any] = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "numerals_format": numerals_format,
        }
        if spoken_form:
            tl_kwargs["spoken_form"] = True
            tl_kwargs["spoken_form_numerals_language"] = spoken_form_numerals_language

        async def api_call() -> Any:
            return await _retry_async(
                self._client.text.transliterate,
                retries=self._retries,
                base_delay=self._retry_base_delay,
                timeout=self._timeout,
                **tl_kwargs,
            )

        def record_cost(elapsed_ms: float, _response: Any) -> None:
            self.costs.record_transliterate(
                len(text),
                call_id=self._active_call_id,
                latency_ms=elapsed_ms,
                metadata={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "numerals_format": numerals_format,
                    "spoken_form": spoken_form,
                    "spoken_form_numerals_language": spoken_form_numerals_language,
                },
            )

        return cast(
            str,
            await self._timed_api_call(
                "Transliteration",
                api_call,
                record_cost,
                lambda r: r.transliterated_text,
                {"src": source_lang, "tgt": target_lang},
            ),
        )


def parse_llm_json(raw: str | None) -> dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown code fences."""
    if not raw:
        logger.warning("LLM returned empty/None content")
        return {"_raw": "", "_parse_error": True}
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    except json.JSONDecodeError:
        for candidate in reversed(_balanced_json_object_candidates(cleaned)):
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    return cast(dict[str, Any], loaded)
            except json.JSONDecodeError:
                # Models commonly emit trailing commas; salvage those.
                try:
                    loaded = json.loads(_TRAILING_COMMA_RE.sub(r"\1", candidate))
                    if isinstance(loaded, dict):
                        return cast(dict[str, Any], loaded)
                except json.JSONDecodeError:
                    continue
    logger.warning("Failed to parse LLM JSON", extra={"raw": raw[:200]})
    return {"_raw": cleaned, "_parse_error": True}


def _balanced_json_object_candidates(text: str) -> list[str]:
    """Return balanced top-level JSON-object substrings found in text."""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None

    return candidates
