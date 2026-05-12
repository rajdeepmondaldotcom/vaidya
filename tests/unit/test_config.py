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

    assert settings.scheme_eval_batch_size == 20
    assert settings.scheme_eval_max_parallel_batches == 3
    assert settings.scheme_retrieval_rank_top_k == 10
