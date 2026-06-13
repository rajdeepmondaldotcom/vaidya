"""Tests for the translation pipeline wrapper.

Covers:
- Short-circuit when source == target language
- Short-circuit for empty / whitespace text
- Successful translation via SarvamClient.translate
- Graceful degradation on translation error
- Forwarding of speaker_gender and output_script params
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.pipeline.translator import Translator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_translator(
    translate_return: str = "translated text",
    translate_side_effect: Exception | None = None,
) -> tuple[Translator, MagicMock]:
    """Create a Translator with a mocked SarvamClient."""
    client = MagicMock()
    if translate_side_effect is not None:
        client.translate = AsyncMock(side_effect=translate_side_effect)
    else:
        client.translate = AsyncMock(return_value=translate_return)
    return Translator(client), client


# ---------------------------------------------------------------------------
# Short-circuit: same language
# ---------------------------------------------------------------------------


class TestTranslateIfNeededShortCircuit:
    @pytest.mark.asyncio
    async def test_same_language_returns_original(self) -> None:
        translator, client = _make_translator()
        result = await translator.translate_if_needed(
            "Namaste", source_lang="hi-IN", target_lang="hi-IN"
        )
        assert result == "Namaste"
        client.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_returns_original(self) -> None:
        translator, client = _make_translator()
        result = await translator.translate_if_needed("", source_lang="en-IN", target_lang="hi-IN")
        assert result == ""
        client.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_text_returns_original(self) -> None:
        translator, client = _make_translator()
        result = await translator.translate_if_needed(
            "   ", source_lang="en-IN", target_lang="hi-IN"
        )
        assert result == "   "
        client.translate.assert_not_called()


# ---------------------------------------------------------------------------
# Successful translation
# ---------------------------------------------------------------------------


class TestTranslateIfNeededSuccess:
    @pytest.mark.asyncio
    async def test_calls_client_translate_and_returns_result(self) -> None:
        translator, client = _make_translator(translate_return="Namaste duniya")
        result = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        assert result == "Namaste duniya"
        client.translate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forwards_source_and_target_lang(self) -> None:
        translator, client = _make_translator(translate_return="Vanakkam")
        await translator.translate_if_needed("Hello", source_lang="en-IN", target_lang="ta-IN")
        call_args = client.translate.call_args
        assert call_args[0][0] == "Hello"
        assert call_args[0][1] == "en-IN"
        assert call_args[0][2] == "ta-IN"


# ---------------------------------------------------------------------------
# Graceful degradation on error
# ---------------------------------------------------------------------------


class TestTranslateIfNeededError:
    @pytest.mark.asyncio
    async def test_returns_original_text_on_translation_error(self) -> None:
        translator, _client = _make_translator(translate_side_effect=RuntimeError("API down"))
        result = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


class TestTranslateIfNeededParams:
    @pytest.mark.asyncio
    async def test_forwards_speaker_gender_and_output_script(self) -> None:
        translator, client = _make_translator(translate_return="result")
        await translator.translate_if_needed(
            "Hello",
            source_lang="en-IN",
            target_lang="hi-IN",
            speaker_gender="Female",
            output_script="roman",
        )
        call_kwargs = client.translate.call_args[1]
        assert call_kwargs["speaker_gender"] == "Female"
        assert call_kwargs["output_script"] == "roman"

    @pytest.mark.asyncio
    async def test_default_speaker_gender_and_output_script(self) -> None:
        translator, client = _make_translator(translate_return="result")
        await translator.translate_if_needed("Hello", source_lang="en-IN", target_lang="hi-IN")
        call_kwargs = client.translate.call_args[1]
        assert call_kwargs["speaker_gender"] == "Male"
        assert call_kwargs["output_script"] == "fully-native"


# ---------------------------------------------------------------------------
# Term preservation
# ---------------------------------------------------------------------------


class TestTermPreservation:
    @pytest.mark.asyncio
    async def test_domain_terms_survive_translation(self) -> None:
        """PM-JAY and Aadhaar should be present in the translated output."""
        translator, client = _make_translator(
            translate_return="[[0]] ke zariye [[1]] se verification"
        )
        result = await translator.translate_if_needed(
            "Get PM-JAY via Aadhaar verification",
            source_lang="en-IN",
            target_lang="hi-IN",
        )
        assert "PM-JAY" in result
        assert "Aadhaar" in result

    @pytest.mark.asyncio
    async def test_case_insensitive_term_matching(self) -> None:
        """Terms should be matched case-insensitively."""
        translator, client = _make_translator(translate_return="Visit [[0]] for [[1]]")
        result = await translator.translate_if_needed(
            "Visit csc for pmjay",
            source_lang="en-IN",
            target_lang="hi-IN",
        )
        # Original case is preserved from the input
        assert "csc" in result.lower() or "CSC" in result

    @pytest.mark.asyncio
    async def test_no_terms_passes_text_unchanged(self) -> None:
        """Text without domain terms should translate normally."""
        translator, client = _make_translator(translate_return="Namaste duniya")
        result = await translator.translate_if_needed(
            "Hello world",
            source_lang="en-IN",
            target_lang="hi-IN",
        )
        assert result == "Namaste duniya"

    @pytest.mark.asyncio
    async def test_protect_term_creates_unique_tokens(self) -> None:
        """_protect_term should create unique numbered tokens."""
        from vaidya.pipeline.translator import Translator

        registry: dict[str, str] = {}
        t1 = Translator._protect_term("PM-JAY", registry)
        t2 = Translator._protect_term("Aadhaar", registry)
        assert t1 == "[[0]]"
        assert t2 == "[[1]]"
        assert registry["[[0]]"] == "PM-JAY"
        assert registry["[[1]]"] == "Aadhaar"

    @pytest.mark.asyncio
    async def test_terms_not_protected_on_same_language(self) -> None:
        """Same-language short-circuit should skip term protection entirely."""
        translator, client = _make_translator()
        result = await translator.translate_if_needed(
            "Visit CSC for PM-JAY",
            source_lang="hi-IN",
            target_lang="hi-IN",
        )
        assert result == "Visit CSC for PM-JAY"
        client.translate.assert_not_called()


# ---------------------------------------------------------------------------
# Translation memoization (LRU cache)
# ---------------------------------------------------------------------------


class TestTranslationCache:
    @pytest.mark.asyncio
    async def test_second_identical_call_hits_cache(self) -> None:
        """A repeated translation returns the cached value without re-calling."""
        translator, client = _make_translator(translate_return="Namaste duniya")

        first = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        second = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )

        assert first == "Namaste duniya"
        assert second == "Namaste duniya"
        # Client invoked exactly once across both calls.
        client.translate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_stores_final_term_preserved_output(self) -> None:
        """Cached value is the post-term-preservation result, not the raw client output."""
        translator, client = _make_translator(
            translate_return="[[0]] ke zariye [[1]] se verification"
        )

        first = await translator.translate_if_needed(
            "Get PM-JAY via Aadhaar verification",
            source_lang="en-IN",
            target_lang="hi-IN",
        )
        second = await translator.translate_if_needed(
            "Get PM-JAY via Aadhaar verification",
            source_lang="en-IN",
            target_lang="hi-IN",
        )

        # Both calls must contain the restored domain terms, and the client
        # is only hit once (the second is served from cache).
        for result in (first, second):
            assert "PM-JAY" in result
            assert "Aadhaar" in result
            assert "[[0]]" not in result
        client.translate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_same_language_does_not_use_cache(self) -> None:
        """src == tgt still short-circuits and never touches the client/cache."""
        translator, client = _make_translator()
        result = await translator.translate_if_needed(
            "Namaste", source_lang="hi-IN", target_lang="hi-IN"
        )
        assert result == "Namaste"
        client.translate.assert_not_called()
        # Short-circuited results are not stored in the cache.
        assert len(translator._cache) == 0

    @pytest.mark.asyncio
    async def test_different_params_are_cached_separately(self) -> None:
        """Different speaker_gender / output_script keys do not collide."""
        translator, client = _make_translator(translate_return="result")

        await translator.translate_if_needed(
            "Hello", source_lang="en-IN", target_lang="hi-IN", speaker_gender="Male"
        )
        await translator.translate_if_needed(
            "Hello", source_lang="en-IN", target_lang="hi-IN", speaker_gender="Female"
        )

        # Distinct keys -> two separate client calls.
        assert client.translate.await_count == 2

    @pytest.mark.asyncio
    async def test_failed_translation_is_not_cached(self) -> None:
        """A failed translation is not cached; a later success recomputes."""
        client = MagicMock()
        client.translate = AsyncMock(side_effect=[RuntimeError("API down"), "Namaste duniya"])
        translator = Translator(client)

        first = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        assert first == "Hello world"  # graceful degradation to original
        assert len(translator._cache) == 0

        second = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        assert second == "Namaste duniya"
        assert client.translate.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_result_retries_within_call(self) -> None:
        """A blank result triggers one immediate retry — never a blank turn."""
        client = MagicMock()
        client.translate = AsyncMock(side_effect=["", "Namaste duniya"])
        translator = Translator(client)

        result = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        assert result == "Namaste duniya"  # retry value, never blank
        assert client.translate.await_count == 2  # original + one retry
        assert len(translator._cache) == 1  # the successful retry IS cached

    @pytest.mark.asyncio
    async def test_blank_after_retry_falls_back_to_original(self) -> None:
        """If both attempts are blank, return the original text (never silence)."""
        client = MagicMock()
        client.translate = AsyncMock(return_value="")
        translator = Translator(client)

        result = await translator.translate_if_needed(
            "Hello world", source_lang="en-IN", target_lang="hi-IN"
        )
        assert result == "Hello world"  # original text, not a blank turn
        assert client.translate.await_count == 2  # tried twice
        assert len(translator._cache) == 0  # blank never cached

    @pytest.mark.asyncio
    async def test_lru_eviction_respects_maxsize(self) -> None:
        """Cache is bounded; the least-recently-used entry is evicted."""
        client = MagicMock()
        client.translate = AsyncMock(side_effect=lambda text, *a, **k: f"tr:{text}")
        translator = Translator(client, cache_maxsize=2)

        await translator.translate_if_needed("a", source_lang="en-IN", target_lang="hi-IN")
        await translator.translate_if_needed("b", source_lang="en-IN", target_lang="hi-IN")
        # Touch "a" so "b" becomes least-recently-used.
        await translator.translate_if_needed("a", source_lang="en-IN", target_lang="hi-IN")
        # Insert "c" -> evicts "b" (LRU). Cache now holds {"a", "c"}.
        await translator.translate_if_needed("c", source_lang="en-IN", target_lang="hi-IN")

        assert len(translator._cache) == 2
        baseline = client.translate.await_count

        # "a" and "c" are still cached: served without new client calls.
        await translator.translate_if_needed("a", source_lang="en-IN", target_lang="hi-IN")
        await translator.translate_if_needed("c", source_lang="en-IN", target_lang="hi-IN")
        assert client.translate.await_count == baseline

        # "b" was evicted: re-translating it calls the client again.
        await translator.translate_if_needed("b", source_lang="en-IN", target_lang="hi-IN")
        assert client.translate.await_count == baseline + 1

    @pytest.mark.asyncio
    async def test_cache_is_per_instance(self) -> None:
        """Two Translator instances do not share a cache."""
        translator_a, client_a = _make_translator(translate_return="A")
        translator_b, client_b = _make_translator(translate_return="B")

        await translator_a.translate_if_needed("Hi", source_lang="en-IN", target_lang="hi-IN")
        await translator_b.translate_if_needed("Hi", source_lang="en-IN", target_lang="hi-IN")

        client_a.translate.assert_awaited_once()
        client_b.translate.assert_awaited_once()
