"""Settings parsing tests."""

from __future__ import annotations

from vaidya.config import Settings


def test_allowed_origins_accepts_single_env_value(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com")

    settings = Settings()

    assert settings.allowed_origins == ["https://example.com"]


def test_allowed_origins_accepts_comma_separated_env_value(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://one.example, https://two.example")

    settings = Settings()

    assert settings.allowed_origins == ["https://one.example", "https://two.example"]


def test_allowed_origins_accepts_json_env_value(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", '["https://one.example","https://two.example"]')

    settings = Settings()

    assert settings.allowed_origins == ["https://one.example", "https://two.example"]


def test_scheme_evaluation_defaults():
    settings = Settings()

    # Small batches + high parallelism is the latency-optimal setting
    # (large batches make one slow call that risks the 45s per-call timeout).
    assert settings.scheme_eval_batch_size == 3
    assert settings.scheme_eval_max_parallel_batches == 8
    assert settings.scheme_retrieval_rank_top_k == 10


def test_conversational_llm_timeout_is_shorter_than_eligibility():
    settings = Settings()

    # Fast conversational calls must fail fast (12s) and never inherit the
    # long 45s eligibility tail, or one hung Sarvam call stalls a simple turn.
    assert settings.conversational_llm_timeout_seconds == 12.0
    assert settings.llm_timeout_seconds == 45.0
    assert settings.conversational_llm_timeout_seconds < settings.llm_timeout_seconds
