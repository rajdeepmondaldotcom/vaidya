"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    chromadb_path: str = "./chroma_data"
    chromadb_host: str = ""
    chromadb_port: int = 8000
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    environment: str = "production"

    # Model routing (see docs.sarvam.ai/api-reference-docs/getting-started/models)
    # Both sarvam-105b and sarvam-30b are FREE. Use 105b for accuracy-critical
    # agents (eligibility, reviewer). Use 30b for latency-sensitive agents
    # (orchestrator, intake, guidance) — 30b is ~3x faster.
    orchestrator_model: str = "sarvam-30b"
    intake_model: str = "sarvam-30b"
    eligibility_model: str = "sarvam-105b"
    reviewer_model: str = "sarvam-105b"
    guidance_model: str = "sarvam-30b"

    # Session
    session_ttl_seconds: int = 1800  # 30 minutes

    # Timeouts. Eligibility + reviewer on the free tier need headroom —
    # they timed out at both 60s and 90s in real calls; the voice edge
    # keeps the caller informed with keepalive notes while this runs.
    agent_timeout_seconds: float = 180.0
    # Per-call ceiling. Eligibility batches on 105b run 10-25s at "low"
    # reasoning; 45s leaves tail headroom so a slow batch retries cleanly
    # instead of timing out mid-flight. Conversational calls finish in ~2s
    # regardless, so this only affects the slow tail.
    llm_timeout_seconds: float = 45.0
    # Fast-path ceiling for CONVERSATIONAL 30b calls (intake, guidance). They
    # normally finish in ~2s, so they must NOT inherit the 45s eligibility tail:
    # a single slow/hung Sarvam call on a simple turn would otherwise stall the
    # caller up to 45s before retrying. 12s fails fast (and retries) while still
    # leaving headroom for a genuinely slow-but-healthy 30b response. Eligibility
    # and reviewer keep the longer llm_timeout_seconds above.
    conversational_llm_timeout_seconds: float = 12.0

    # Advanced LLM. sarvam-30b/105b are ALWAYS-ON reasoning models: the API
    # only accepts reasoning_effort in {low, medium, high} (omitting it ->
    # verbose default), and the free tier caps max_tokens at 4096. Reasoning
    # consumes that budget first, so a call only emits clean JSON content if
    # reasoning fits well under 4096. "low" is the floor and is what keeps
    # both conversational AND scheme-eval turns from truncating to empty
    # content (the cause of the multi-minute no-result failures).
    eligibility_reasoning_effort: str = "low"
    reviewer_reasoning_effort: str = "low"
    intake_reasoning_effort: str = "low"
    guidance_reasoning_effort: str = "low"
    wiki_grounding: bool = True

    # Scheme evaluation. Keep batches SMALL and parallelism HIGH — this is a
    # LATENCY optimisation, not just a token-budget one. Many small 105b calls
    # run concurrently and finish in ~one call's time; one big batch is a single
    # long call that also risks the per-call llm_timeout (45s) and then retries.
    # Measured on the paid tier: batch_size=10 blew a 6-turn sim past 220s,
    # whereas batch_size=3 keeps it near real-time. The terse prompt output (no
    # per-scheme reasoning trace) keeps each batch small.
    scheme_eval_batch_size: int = 3
    scheme_eval_max_parallel_batches: int = 8
    scheme_retrieval_rank_top_k: int = 10

    # Language & translation
    default_language: str = "hi-IN"
    translate_model_voice: str = "mayura:v1"  # colloquial, 11 voice languages
    translate_model_text: str = "sarvam-translate:v1"  # formal, 23 languages

    # TTS
    tts_model: str = "bulbul:v3"
    tts_sample_rate: int = 8000
    tts_default_pace: float = 0.94
    tts_repair_pace: float = 0.88
    tts_distress_pace: float = 0.85
    tts_results_pace: float = 0.92
    tts_temperature: float = 0.55
    tts_min_buffer_size: int = 35
    tts_max_chunk_length: int = 130

    # STT
    stt_model: str = "saaras:v3"
    stt_mode: str = "transcribe"
    voice_stt_mode: str = "codemix"
    stt_interrupt_min_speech_frames: int = 3

    # Resilience
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 30.0
    rate_limit_per_minute: int = 60
    redis_max_connections: int = 10

    # Security
    allowed_origins: Annotated[list[str], NoDecode] = []

    # Monitoring
    sentry_dsn: str = ""

    # Telephony (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    voice_websocket_url: str = ""
    voice_status_callback_url: str = ""
    default_voice_language: str = "hi-IN"
    telephony_provider: str = "twilio"
    telephony_rate_inr_per_minute: float = 0.0

    # Simulation
    max_simulation_turns: int = 20

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: Any) -> list[str]:
        """Accept JSON, comma-separated, or single-origin env values."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    return [str(item).strip() for item in decoded if str(item).strip()]
                raise ValueError("ALLOWED_ORIGINS JSON value must be a list")
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        raise TypeError("ALLOWED_ORIGINS must be a string or list")
