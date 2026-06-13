"""Bounded in-memory cache for synthesized TTS audio.

Fixed/templated voice prompts — the greeting, the five intake questions, the
processing filler, silence nudges, and the closure line — are byte-identical on
every call, yet each call re-synthesizes them through Sarvam Bulbul. That round
trip is the dominant per-turn latency for those prompts. Caching the audio
bytes keyed on everything that affects the waveform makes the second (and every
later) render a dictionary lookup instead of a network call.

The cache wraps a :class:`~vaidya.sarvam.client.SarvamClient` and falls back to
``client.tts()`` on a miss, so the circuit breaker, retries, and cost tracking
in the client are preserved unchanged. Failures (``None`` results) are never
cached — a transient TTS error must not poison a fixed prompt for the rest of
the process lifetime.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)

# Default bound on distinct cached entries. Fixed prompts number in the low
# dozens per (language, pace) combination; across 11 voice languages and a
# handful of paces this stays comfortably under a few hundred. 512 leaves
# generous headroom while still capping memory if exact-text caching ever
# admits unexpected dynamic strings.
DEFAULT_MAX_ENTRIES = 512

# Cache key tuple: (text, language, pace, speaker, model, sample_rate).
CacheKey = tuple[str, str, float, str, str, int]


class TTSCache:
    """LRU-bounded cache of synthesized audio over a :class:`SarvamClient`.

    The key includes every argument that changes the synthesized waveform:
    ``text``, ``language``, ``pace``, ``speaker``, ``model``, and
    ``sample_rate``. Two prompts that differ in pace (e.g. the calm intake pace
    vs. the slower repair pace) are cached separately, as Sarvam returns
    different audio for each.
    """

    def __init__(self, client: SarvamClient, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._client = client
        # max_entries <= 0 would make every store evict immediately; clamp to 1
        # so the cache is always at least functional.
        self._max_entries = max(1, max_entries)
        self._cache: OrderedDict[CacheKey, bytes] = OrderedDict()
        # Guards the OrderedDict against interleaved coroutine mutation. The
        # render itself happens outside the lock so concurrent misses for
        # different keys don't serialize on the network call.
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get_or_render(
        self,
        text: str,
        language: str,
        *,
        pace: float,
        speaker: str = "priya",
        model: str = "bulbul:v3",
        sample_rate: int = 8000,
    ) -> bytes | None:
        """Return cached audio for the prompt, else synthesize and cache it.

        On a cache hit the stored bytes are returned without calling the
        client. On a miss ``client.tts()`` is invoked with the same arguments;
        a non-``None`` result is stored and returned, while ``None`` (a TTS
        error or open circuit) is returned without being cached so the next
        attempt re-tries.
        """
        key: CacheKey = (text, language, pace, speaker, model, sample_rate)

        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return cached
            self._misses += 1

        # Render outside the lock: the network round trip must not block hits
        # or misses for other keys. A concurrent duplicate miss may render the
        # same prompt twice, but the result is identical and the last store
        # wins — correctness is preserved and the (rare) double render is
        # cheaper than serializing every synthesis.
        audio = await self._client.tts(
            text,
            language,
            speaker=speaker,
            model=model,
            pace=pace,
            speech_sample_rate=sample_rate,
        )

        if audio is None:
            # Never cache a failure — it would freeze a transient error onto a
            # fixed prompt for the life of the process.
            return None

        async with self._lock:
            self._cache[key] = audio
            self._cache.move_to_end(key)
            self._evict_if_needed()
        return audio

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used entries until within the bound.

        Caller must hold ``self._lock``.
        """
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    async def warm(self, items: Iterable[Mapping[str, object]]) -> int:
        """Pre-render known fixed prompts so the first call is already a hit.

        Each item is a mapping with at least ``text`` and ``language`` keys and
        optional ``pace`` / ``speaker`` / ``model`` / ``sample_rate`` overrides.
        Best-effort and guarded: any failure is logged and skipped so a TTS
        outage at startup never crashes the lifespan. Returns the number of
        prompts successfully rendered and cached.
        """
        warmed = 0
        for item in items:
            try:
                text = str(item["text"])
                language = str(item["language"])
                pace = float(item.get("pace", 1.0))  # type: ignore[arg-type]
                speaker = str(item.get("speaker", "priya"))
                model = str(item.get("model", "bulbul:v3"))
                sample_rate = int(item.get("sample_rate", 8000))  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed TTS warm item: %r", item)
                continue

            try:
                audio = await self.get_or_render(
                    text,
                    language,
                    pace=pace,
                    speaker=speaker,
                    model=model,
                    sample_rate=sample_rate,
                )
            except Exception as exc:  # noqa: BLE001 - warm must never crash startup
                logger.warning(
                    "TTS warm render failed",
                    extra={"language": language, "error": str(exc)[:200]},
                )
                continue
            if audio is not None:
                warmed += 1

        if warmed:
            logger.info("TTS cache warmed with %d fixed prompts", warmed)
        return warmed

    @property
    def stats(self) -> dict[str, int]:
        """Return hit/miss/size counters for observability."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}

    def __len__(self) -> int:
        return len(self._cache)
