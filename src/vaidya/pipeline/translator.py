"""Translation wrapper for the Vaidya conversation pipeline.

Provides a single :meth:`translate_if_needed` method that short-circuits
when source and target languages match, avoiding unnecessary API calls, and
memoizes identical translations in a bounded per-instance LRU cache so that
repeated round-trips of the same string skip the network entirely.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from vaidya.pipeline.translation_terms import PRESERVE_RE
from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)

# Cache key: (text, source_lang, target_lang, speaker_gender, output_script).
# All five inputs can change the final translated string, so all are keyed.
_CacheKey = tuple[str, str, str, str, str]

# Bound on the per-instance translation cache. Identical per-turn round-trips
# (e.g. repeated prompts / responses) are served from memory instead of
# re-hitting the Sarvam API.
DEFAULT_CACHE_MAXSIZE = 512


class Translator:
    """Thin wrapper around :meth:`SarvamClient.translate`.

    The wrapper exists so the pipeline can inject translation as a
    composable step without coupling to the Sarvam client's full
    interface.

    A bounded in-process LRU cache memoizes identical translations so that
    repeated round-trips of the same string skip the network call. The cache
    is per-instance and keyed by every input that affects the final output.
    """

    def __init__(self, client: SarvamClient, cache_maxsize: int = DEFAULT_CACHE_MAXSIZE) -> None:
        self._client = client
        # OrderedDict acts as an LRU: most-recently-used entries move to the
        # end, and we evict from the front once we exceed ``cache_maxsize``.
        self._cache: OrderedDict[_CacheKey, str] = OrderedDict()
        self._cache_maxsize = max(0, cache_maxsize)

    async def translate_if_needed(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        speaker_gender: str = "Male",
        output_script: str = "fully-native",
    ) -> str:
        """Translate *text* from *source_lang* to *target_lang*.

        Returns *text* unchanged when the two languages match.  On
        translation failure, returns the original text and logs the error
        (degrading gracefully rather than failing the turn).

        Identical successful translations are served from a bounded
        per-instance LRU cache keyed by ``(text, source_lang, target_lang,
        speaker_gender, output_script)``; empty or failed results are never
        cached.

        Parameters
        ----------
        text:
            The text to translate.
        source_lang:
            BCP-47 source language code (e.g. ``"en-IN"``).
        target_lang:
            BCP-47 target language code (e.g. ``"hi-IN"``).
        speaker_gender:
            Gender hint for gendered translations (default ``"Male"``).
        output_script:
            Script preference for the output (default ``"fully-native"``).

        Returns
        -------
        str
            Translated text, or the original on same-language / error.
        """
        if not text or not text.strip():
            return text

        if source_lang == target_lang:
            return text

        cache_key: _CacheKey = (text, source_lang, target_lang, speaker_gender, output_script)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(
                "Translation cache hit",
                extra={"source_lang": source_lang, "target_lang": target_lang},
            )
            return cached

        try:
            # Preserve domain terms through the translation round-trip
            preserved: dict[str, str] = {}
            protected = PRESERVE_RE.sub(lambda m: self._protect_term(m.group(), preserved), text)

            translated = await self._client.translate(
                protected,
                source_lang,
                target_lang,
                speaker_gender=speaker_gender,
                output_script=output_script,
            )

            # Sarvam occasionally returns a blank string under load. Retry once —
            # a blank reply is the worst outcome (the caller hears silence on that
            # turn); an untranslated line is at least something.
            if not (translated and translated.strip()):
                logger.warning(
                    "Empty translation result, retrying once",
                    extra={"source_lang": source_lang, "target_lang": target_lang},
                )
                translated = await self._client.translate(
                    protected,
                    source_lang,
                    target_lang,
                    speaker_gender=speaker_gender,
                    output_script=output_script,
                )

            # Restore preserved terms
            for token, original in preserved.items():
                translated = translated.replace(token, original)

            logger.debug(
                "Translation completed",
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "input_length": len(text),
                    "output_length": len(translated),
                    "terms_preserved": len(preserved),
                },
            )

            # Never hand back a blank turn: fall back to the original text when the
            # translation is still empty after the retry. Only non-empty results
            # are memoised (blank/failed are retried on the next turn).
            if translated and translated.strip():
                self._cache_set(cache_key, translated)
                return translated

            logger.error(
                "Translation blank after retry, using original text",
                extra={"source_lang": source_lang, "target_lang": target_lang},
            )
            return text
        except Exception as exc:
            logger.error(
                "Translation failed, returning original text",
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "error": str(exc),
                },
            )
            return text

    @staticmethod
    def _protect_term(term: str, registry: dict[str, str]) -> str:
        """Replace *term* with a unique token and record the mapping."""
        # Letter-free token: Sarvam's fully-native output transliterates Latin
        # letters, so "__TERM0__" came back as "टर्म0"/"டெர்ம்0"/"টার্ম0" and the
        # restore below missed it (placeholder leaked into speech). Brackets +
        # digits pass through every script verbatim (probed on hi/ta/bn), so the
        # exact-match restore always succeeds. Distinct non-overlapping tokens
        # ("[[1]]" is not a substring of "[[10]]"), so restore order is safe.
        token = f"[[{len(registry)}]]"
        registry[token] = term
        return token

    def _cache_get(self, key: _CacheKey) -> str | None:
        """Return the cached translation for *key*, marking it most-recent.

        Returns ``None`` on a miss (or when caching is disabled).
        """
        if self._cache_maxsize <= 0:
            return None
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _cache_set(self, key: _CacheKey, value: str) -> None:
        """Store *value* for *key*, evicting the least-recently-used entry."""
        if self._cache_maxsize <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)
