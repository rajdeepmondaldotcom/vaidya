"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Timeouts
    agent_timeout_seconds: float = 60.0
    llm_timeout_seconds: float = 30.0

    # Advanced LLM
    eligibility_reasoning_effort: str = "high"
    reviewer_reasoning_effort: str = "high"
    intake_reasoning_effort: str = "low"
    guidance_reasoning_effort: str = "low"
    wiki_grounding: bool = True

    # Language & translation
    default_language: str = "hi-IN"
    translate_model_voice: str = "mayura:v1"  # colloquial, 11 voice languages
    translate_model_text: str = "sarvam-translate:v1"  # formal, 23 languages

    # TTS
    tts_sample_rate: int = 8000
    tts_default_pace: float = 1.0
    tts_distress_pace: float = 0.85

    # STT
    stt_model: str = "saaras:v3"
    stt_mode: str = "transcribe"

    # Resilience
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 30.0
    rate_limit_per_minute: int = 60
    redis_max_connections: int = 10

    # Security
    allowed_origins: list[str] = []

    # Monitoring
    sentry_dsn: str = ""

    # Telephony (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    voice_websocket_url: str = ""
    voice_status_callback_url: str = ""
    default_voice_language: str = "hi-IN"

    # Simulation
    max_simulation_turns: int = 20
