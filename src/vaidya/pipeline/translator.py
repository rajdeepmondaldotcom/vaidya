"""Translation wrapper for the Vaidya conversation pipeline.

Provides a single :meth:`translate_if_needed` method that short-circuits
when source and target languages match, avoiding unnecessary API calls.
"""

from __future__ import annotations

import logging

from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)


class Translator:
    """Thin wrapper around :meth:`SarvamClient.translate`.

    The wrapper exists so the pipeline can inject translation as a
    composable step without coupling to the Sarvam client's full
    interface.
    """

    def __init__(self, client: SarvamClient) -> None:
        self._client = client

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

        try:
            translated = await self._client.translate(
                text,
                source_lang,
                target_lang,
                speaker_gender=speaker_gender,
                output_script=output_script,
            )
            logger.debug(
                "Translation completed",
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "input_length": len(text),
                    "output_length": len(translated),
                },
            )
            return translated
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
