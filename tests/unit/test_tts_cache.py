"""Unit tests for the bounded TTS audio cache."""

from __future__ import annotations

from unittest.mock import AsyncMock

from vaidya.sarvam.tts_cache import DEFAULT_MAX_ENTRIES, TTSCache


def _client(return_value: bytes | None = b"\x00\x01\x02") -> AsyncMock:
    """A mock SarvamClient whose tts() is an AsyncMock returning audio bytes."""
    client = AsyncMock()
    client.tts = AsyncMock(return_value=return_value)
    return client


async def test_miss_calls_client_once_and_returns_bytes() -> None:
    """A cache miss synthesizes via client.tts and returns its bytes."""
    client = _client(b"audio-1")
    cache = TTSCache(client)

    result = await cache.get_or_render("Namaste", "hi-IN", pace=0.94)

    assert result == b"audio-1"
    client.tts.assert_awaited_once()
    # The render must carry the key-defining kwargs through to the client.
    _, kwargs = client.tts.call_args
    assert kwargs["pace"] == 0.94
    assert kwargs["speech_sample_rate"] == 8000


async def test_second_identical_call_is_cached_without_calling_client() -> None:
    """A repeated identical request returns cached bytes, no second tts call."""
    client = _client(b"audio-1")
    cache = TTSCache(client)

    first = await cache.get_or_render("Namaste", "hi-IN", pace=0.94)
    second = await cache.get_or_render("Namaste", "hi-IN", pace=0.94)

    assert first == second == b"audio-1"
    client.tts.assert_awaited_once()  # still only the first (miss) render
    assert cache.stats == {"hits": 1, "misses": 1, "size": 1}


async def test_none_result_is_not_cached() -> None:
    """A None (error/open-circuit) result is returned but never cached."""
    client = _client(None)
    cache = TTSCache(client)

    first = await cache.get_or_render("Namaste", "hi-IN", pace=0.94)
    second = await cache.get_or_render("Namaste", "hi-IN", pace=0.94)

    assert first is None
    assert second is None
    # Both calls miss and re-render because None is never stored.
    assert client.tts.await_count == 2
    assert len(cache) == 0


async def test_none_then_success_caches_the_success() -> None:
    """Once a render finally succeeds, it is cached and stops calling tts."""
    client = AsyncMock()
    client.tts = AsyncMock(side_effect=[None, b"audio-late"])
    cache = TTSCache(client)

    first = await cache.get_or_render("Bas thoda aur", "hi-IN", pace=0.92)
    second = await cache.get_or_render("Bas thoda aur", "hi-IN", pace=0.92)
    third = await cache.get_or_render("Bas thoda aur", "hi-IN", pace=0.92)

    assert first is None
    assert second == b"audio-late"
    assert third == b"audio-late"
    assert client.tts.await_count == 2  # miss, then miss-that-succeeded; third hits


async def test_key_includes_pace_language_and_speaker() -> None:
    """Differing pace / language / speaker are distinct cache entries."""
    client = _client(b"audio")
    cache = TTSCache(client)

    await cache.get_or_render("Q1", "hi-IN", pace=0.94)
    await cache.get_or_render("Q1", "hi-IN", pace=0.88)  # different pace
    await cache.get_or_render("Q1", "ta-IN", pace=0.94)  # different language
    await cache.get_or_render("Q1", "hi-IN", pace=0.94, speaker="anushka")  # speaker

    assert client.tts.await_count == 4
    assert len(cache) == 4


async def test_lru_eviction_bounds_the_cache() -> None:
    """Storing beyond max_entries evicts the least-recently-used entry."""
    client = _client(b"audio")
    cache = TTSCache(client, max_entries=2)

    await cache.get_or_render("a", "hi-IN", pace=1.0)
    await cache.get_or_render("b", "hi-IN", pace=1.0)
    # Touch "a" so "b" becomes the LRU entry.
    await cache.get_or_render("a", "hi-IN", pace=1.0)
    # Insert "c" -> should evict "b", not "a".
    await cache.get_or_render("c", "hi-IN", pace=1.0)

    assert len(cache) == 2
    client.tts.reset_mock()
    # "a" and "c" are still cached (no new render); "b" was evicted (re-renders).
    await cache.get_or_render("a", "hi-IN", pace=1.0)
    await cache.get_or_render("c", "hi-IN", pace=1.0)
    client.tts.assert_not_awaited()
    await cache.get_or_render("b", "hi-IN", pace=1.0)
    client.tts.assert_awaited_once()


async def test_max_entries_clamped_to_at_least_one() -> None:
    """A non-positive max_entries is clamped so the cache still functions."""
    client = _client(b"audio")
    cache = TTSCache(client, max_entries=0)

    result = await cache.get_or_render("a", "hi-IN", pace=1.0)
    assert result == b"audio"
    assert len(cache) == 1


async def test_warm_prerenders_items_and_survives_failures() -> None:
    """warm() renders each item, skips failures, and reports the success count."""
    client = AsyncMock()
    # Second item renders None (failure); others succeed.
    client.tts = AsyncMock(side_effect=[b"a", None, b"c"])
    cache = TTSCache(client)

    warmed = await cache.warm(
        [
            {"text": "greeting", "language": "hi-IN", "pace": 0.94},
            {"text": "q1", "language": "hi-IN", "pace": 0.94},
            {"text": "closure", "language": "ta-IN", "pace": 0.92, "speaker": "anushka"},
        ]
    )

    assert warmed == 2  # None result not counted
    assert client.tts.await_count == 3
    # Re-rendering a warmed item is a hit (no extra tts call).
    client.tts.reset_mock()
    again = await cache.get_or_render("greeting", "hi-IN", pace=0.94)
    assert again == b"a"
    client.tts.assert_not_awaited()


async def test_warm_skips_malformed_items() -> None:
    """warm() ignores items missing required keys without raising."""
    client = _client(b"audio")
    cache = TTSCache(client)

    warmed = await cache.warm(
        [
            {"language": "hi-IN"},  # missing text
            {"text": "ok", "language": "hi-IN"},
        ]
    )

    assert warmed == 1
    client.tts.assert_awaited_once()


def test_default_max_entries_is_positive() -> None:
    """The module default bound is a sane positive number."""
    assert DEFAULT_MAX_ENTRIES > 0
