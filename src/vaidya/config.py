"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    sarvam_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    chromadb_path: str = "./chroma_data"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"

    # Model routing (see docs.sarvam.ai/api-reference-docs/getting-started/models)
    # Both sarvam-105b and sarvam-30b are FREE. Use 105b everywhere for
    # maximum accuracy and JSON compliance. Switch to 30b only if latency
    # becomes a bottleneck in production (30b is ~3x faster).
    orchestrator_model: str = "sarvam-105b"
    intake_model: str = "sarvam-105b"
    eligibility_model: str = "sarvam-105b"
    reviewer_model: str = "sarvam-105b"
    guidance_model: str = "sarvam-105b"

    # Session
    session_ttl_seconds: int = 1800  # 30 minutes

    # Timeouts
    agent_timeout_seconds: float = 15.0
    llm_timeout_seconds: float = 10.0

    # Advanced LLM
    eligibility_reasoning_effort: str = "high"
    reviewer_reasoning_effort: str = "high"
    intake_reasoning_effort: str = "low"
    guidance_reasoning_effort: str = "low"
    wiki_grounding: bool = True

    # TTS
    tts_sample_rate: int = 8000
    tts_default_pace: float = 1.0
    tts_distress_pace: float = 0.85

    # STT
    stt_model: str = "saaras:v3"
    stt_mode: str = "transcribe"
