"""Tests for language detection, normalisation, and constants.

Covers:
- normalize_language with short codes, BCP-47 tags, full names
- Case insensitivity and whitespace handling
- Unrecognised input defaults to Language.HINDI
- is_supported for valid and invalid codes
- get_sarvam_code returns BCP-47 value for each Language
- TTS_SPEAKERS and LANGUAGE_DISPLAY_NAMES completeness
"""

from __future__ import annotations

from vaidya.voice.language import (
    LANGUAGE_DISPLAY_NAMES,
    TTS_SPEAKERS,
    Language,
    detect_script_language,
    get_sarvam_code,
    is_filler_utterance,
    is_supported,
    looks_like_english,
    normalize_language,
)

# ---------------------------------------------------------------------------
# normalize_language: short ISO codes
# ---------------------------------------------------------------------------


class TestNormalizeLanguageShortCodes:
    def test_hi(self) -> None:
        assert normalize_language("hi") is Language.HINDI

    def test_ta(self) -> None:
        assert normalize_language("ta") is Language.TAMIL

    def test_bn(self) -> None:
        assert normalize_language("bn") is Language.BENGALI

    def test_en(self) -> None:
        assert normalize_language("en") is Language.ENGLISH


# ---------------------------------------------------------------------------
# normalize_language: BCP-47 tags
# ---------------------------------------------------------------------------


class TestNormalizeLanguageBCP47:
    def test_hi_in(self) -> None:
        assert normalize_language("hi-IN") is Language.HINDI

    def test_ta_in(self) -> None:
        assert normalize_language("ta-IN") is Language.TAMIL

    def test_bn_in(self) -> None:
        assert normalize_language("bn-IN") is Language.BENGALI

    def test_en_in(self) -> None:
        assert normalize_language("en-IN") is Language.ENGLISH


# ---------------------------------------------------------------------------
# normalize_language: full names
# ---------------------------------------------------------------------------


class TestNormalizeLanguageFullNames:
    def test_hindi(self) -> None:
        assert normalize_language("hindi") is Language.HINDI

    def test_tamil(self) -> None:
        assert normalize_language("tamil") is Language.TAMIL

    def test_bengali(self) -> None:
        assert normalize_language("bengali") is Language.BENGALI

    def test_bangla(self) -> None:
        assert normalize_language("bangla") is Language.BENGALI

    def test_english(self) -> None:
        assert normalize_language("english") is Language.ENGLISH


# ---------------------------------------------------------------------------
# normalize_language: case insensitivity
# ---------------------------------------------------------------------------


class TestNormalizeLanguageCaseInsensitive:
    def test_uppercase_bcp47(self) -> None:
        assert normalize_language("HI-IN") is Language.HINDI

    def test_titlecase_full_name(self) -> None:
        assert normalize_language("Hindi") is Language.HINDI

    def test_uppercase_full_name(self) -> None:
        assert normalize_language("TAMIL") is Language.TAMIL


# ---------------------------------------------------------------------------
# normalize_language: whitespace handling
# ---------------------------------------------------------------------------


class TestNormalizeLanguageWhitespace:
    def test_leading_trailing_whitespace(self) -> None:
        assert normalize_language("  hi  ") is Language.HINDI

    def test_whitespace_around_bcp47(self) -> None:
        assert normalize_language(" ta-IN ") is Language.TAMIL


# ---------------------------------------------------------------------------
# normalize_language: unrecognised input defaults to HINDI
# ---------------------------------------------------------------------------


class TestNormalizeLanguageDefault:
    def test_unrecognised_code_defaults_to_hindi(self) -> None:
        assert normalize_language("xx") is Language.HINDI

    def test_garbage_input_defaults_to_hindi(self) -> None:
        assert normalize_language("martian") is Language.HINDI


# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------


class TestIsSupported:
    def test_returns_true_for_valid_short_code(self) -> None:
        assert is_supported("hi") is True

    def test_returns_true_for_valid_bcp47(self) -> None:
        assert is_supported("ta-IN") is True

    def test_returns_true_for_valid_full_name(self) -> None:
        assert is_supported("bengali") is True

    def test_returns_false_for_unsupported(self) -> None:
        assert is_supported("xx") is False

    def test_returns_false_for_empty_string(self) -> None:
        assert is_supported("") is False


# ---------------------------------------------------------------------------
# get_sarvam_code
# ---------------------------------------------------------------------------


class TestGetSarvamCode:
    def test_hindi_returns_bcp47(self) -> None:
        assert get_sarvam_code(Language.HINDI) == "hi-IN"

    def test_tamil_returns_bcp47(self) -> None:
        assert get_sarvam_code(Language.TAMIL) == "ta-IN"

    def test_bengali_returns_bcp47(self) -> None:
        assert get_sarvam_code(Language.BENGALI) == "bn-IN"

    def test_english_returns_bcp47(self) -> None:
        assert get_sarvam_code(Language.ENGLISH) == "en-IN"


# ---------------------------------------------------------------------------
# TTS_SPEAKERS completeness
# ---------------------------------------------------------------------------


# Speakers accepted by BOTH bulbul:v3 surfaces, taken verbatim from each
# API's 400 error listing (the two lists differ: "anushka"/"abhilash" are
# streaming-only, "niharika" is REST-only, "amelia" is v2-only). A speaker
# outside this intersection 400s on one surface and the bot goes silent.
_BULBUL_V3_SPEAKERS = frozenset(
    {
        "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja",
        "rohan", "simran", "kavya", "amit", "dev", "ishita", "shreya",
        "ratan", "varun", "manan", "sumit", "roopa", "kabir", "aayan",
        "shubh", "advait", "anand", "tanya", "tarun", "sunny", "mani",
        "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan",
        "soham", "rupali",
    }
)  # fmt: skip


class TestTTSSpeakers:
    def test_has_entry_for_every_language(self) -> None:
        for lang in Language:
            assert lang in TTS_SPEAKERS, f"Missing TTS speaker for {lang}"

    def test_values_are_non_empty_strings(self) -> None:
        for lang, speaker in TTS_SPEAKERS.items():
            assert isinstance(speaker, str) and speaker, f"Empty speaker for {lang}"

    def test_every_speaker_is_valid_for_bulbul_v3(self) -> None:
        """An invalid speaker 400s at call time and the bot goes silent."""
        for lang, speaker in TTS_SPEAKERS.items():
            assert speaker in _BULBUL_V3_SPEAKERS, (
                f"{lang}: speaker {speaker!r} is not a bulbul:v3 voice"
            )


# ---------------------------------------------------------------------------
# is_filler_utterance
# ---------------------------------------------------------------------------


class TestIsFillerUtterance:
    def test_common_fillers_across_languages(self) -> None:
        for utterance in ["Okay", "ok!", "Hello", "हाँ", "ठीक है", "ঠিক আছে", "Haan ji", "hmm"]:
            assert is_filler_utterance(utterance), utterance

    def test_substantive_utterances_are_not_fillers(self) -> None:
        for utterance in [
            "Tamil",
            "Main Bihar mein rehta hoon",
            "আমার কি কি স্বাস্থ্য প্রকল্প আছে?",
            "My family has five people",
        ]:
            assert not is_filler_utterance(utterance), utterance

    def test_empty_and_whitespace_are_not_fillers(self) -> None:
        assert not is_filler_utterance("")
        assert not is_filler_utterance("   ")


# ---------------------------------------------------------------------------
# LANGUAGE_DISPLAY_NAMES completeness
# ---------------------------------------------------------------------------


class TestLanguageDisplayNames:
    def test_has_entry_for_every_language(self) -> None:
        for lang in Language:
            assert lang in LANGUAGE_DISPLAY_NAMES, f"Missing display name for {lang}"

    def test_values_are_non_empty_strings(self) -> None:
        for lang, name in LANGUAGE_DISPLAY_NAMES.items():
            assert isinstance(name, str) and name, f"Empty display name for {lang}"


# ---------------------------------------------------------------------------
# detect_script_language: the caller's language from the transcript's script
# (regression guard for the real call where Saaras tagged Bengali as en-IN and
# the bot replied in English -- the script must override the wrong tag)
# ---------------------------------------------------------------------------


class TestDetectScriptLanguage:
    def test_bengali_script(self) -> None:
        assert detect_script_language("বাড়িতে চারজন আছি") is Language.BENGALI

    def test_bengali_full_sentence(self) -> None:
        assert detect_script_language("আমি দিন মজুরি করি") is Language.BENGALI

    def test_devanagari_maps_to_hindi(self) -> None:
        assert detect_script_language("मैं उत्तर प्रदेश में रहता हूँ") is Language.HINDI

    def test_tamil_script(self) -> None:
        assert detect_script_language("நான் தமிழ் பேசுகிறேன்") is Language.TAMIL

    def test_telugu_script(self) -> None:
        assert detect_script_language("నేను తెలుగు మాట్లాడతాను") is Language.TELUGU

    def test_gujarati_script(self) -> None:
        assert detect_script_language("હું ગુજરાતી બોલું છું") is Language.GUJARATI

    def test_punjabi_gurmukhi(self) -> None:
        assert detect_script_language("ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ") is Language.PUNJABI

    def test_dominant_script_wins_over_stray_latin(self) -> None:
        # Saaras often leaves a place name in Latin inside Bengali script.
        assert detect_script_language("আমি Howrah তে থাকি") is Language.BENGALI

    def test_romanized_is_none(self) -> None:
        # The exact first answer that broke the real call (romanized, no script).
        assert detect_script_language("Ami Paschim Banga thaki Howrah") is None

    def test_plain_english_is_none(self) -> None:
        assert detect_script_language("I live in West Bengal") is None

    def test_empty_is_none(self) -> None:
        assert detect_script_language("") is None


# ---------------------------------------------------------------------------
# looks_like_english: gate for honouring a bare en-IN STT tag
# ---------------------------------------------------------------------------


class TestLooksLikeEnglish:
    def test_clear_english_sentence(self) -> None:
        assert looks_like_english("I want to know what schemes are there") is True

    def test_english_about_family(self) -> None:
        assert looks_like_english("There are five people in my family") is True

    def test_romanized_bengali_is_not_english(self) -> None:
        # The real-call first answer must NOT be read as English.
        assert looks_like_english("Ami Paschim Banga thaki Howrah") is False

    def test_romanized_hindi_is_not_english(self) -> None:
        assert looks_like_english("main UP me rehta hoon") is False

    def test_short_filler_is_not_english(self) -> None:
        assert looks_like_english("haan") is False

    def test_single_marker_is_not_enough(self) -> None:
        # One ambiguous hit shouldn't flip a regional answer to English.
        assert looks_like_english("Howrah live") is False
