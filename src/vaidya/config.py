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
    # sarvam-105b: flagship 128K context, best accuracy (free tier)
    # sarvam-30b: efficient 64K context, good for simple tasks (free tier)
    orchestrator_model: str = "sarvam-30b"  # Simple routing classification
    intake_model: str = "sarvam-30b"  # Structured Q&A, low complexity
    eligibility_model: str = "sarvam-105b"  # Complex reasoning + RAG matching
    reviewer_model: str = "sarvam-105b"  # Independent validation, must match eligibility
    guidance_model: str = "sarvam-30b"  # Template-based output generation

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
