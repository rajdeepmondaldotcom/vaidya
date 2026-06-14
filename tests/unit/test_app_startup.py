"""Smoke test that create_app() and lifespan wiring work without crashes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_create_app_returns_fastapi_instance():
    """create_app() should return a FastAPI app with routes registered."""
    from vaidya.app import create_app

    app = create_app()
    assert app.title == "Vaidya"

    # Check the OpenAPI schema, not app.routes: it's the version-stable list of
    # registered endpoints. (Newer FastAPI in our supported range wraps
    # include_router results in opaque route objects, so walking app.routes is
    # brittle, but the generated schema always reflects every included router.)
    paths = app.openapi()["paths"]
    assert "/health" in paths
    assert "/ready" in paths
    assert "/costs" in paths


def test_create_app_has_middleware():
    """The app should have RequestIdMiddleware and CORSMiddleware."""
    from vaidya.app import create_app

    app = create_app()
    middleware_classes = [m.cls.__name__ for m in app.user_middleware]
    assert "RequestIdMiddleware" in middleware_classes
    assert "CORSMiddleware" in middleware_classes


@pytest.mark.asyncio
async def test_lifespan_wires_conversation_manager():
    """Verify the lifespan creates a ConversationManager with correct params.

    We mock all external services (Redis, Chroma, Sarvam) so the test
    doesn't require any infrastructure.
    """
    mock_redis = AsyncMock()
    mock_redis.close = AsyncMock()

    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    attrs = dict(
        sarvam_api_key="test-key",
        chromadb_path="./test_chroma",
        chromadb_host="",
        chromadb_port=8000,
        redis_url="redis://localhost:6379/0",
        session_ttl_seconds=1800,
        host="0.0.0.0",
        port=8000,
        log_level="WARNING",
        intake_model="sarvam-30b",
        eligibility_model="sarvam-105b",
        reviewer_model="sarvam-105b",
        guidance_model="sarvam-30b",
        orchestrator_model="sarvam-30b",
        agent_timeout_seconds=15.0,
        llm_timeout_seconds=45.0,
        conversational_llm_timeout_seconds=12.0,
        scheme_eval_batch_size=20,
        scheme_eval_max_parallel_batches=3,
        scheme_retrieval_rank_top_k=10,
        environment="test",
    )
    mock_settings = MagicMock(**attrs)

    with (
        patch("vaidya.config.Settings", return_value=mock_settings),
        patch("vaidya.sarvam.client.SarvamAI"),
        patch("vaidya.session.manager.SessionManager", return_value=mock_redis),
        patch("vaidya.knowledge.store.chromadb") as mock_chromadb,
        patch("vaidya.knowledge.loader.load_schemes_into_store", return_value=8),
    ):
        mock_chromadb.PersistentClient.return_value = mock_chroma_client

        from vaidya.app import create_app

        app = create_app()

        async with app.router.lifespan_context(app):
            assert hasattr(app.state, "conversation_manager")
            assert hasattr(app.state, "client")
            assert hasattr(app.state, "consent_tracker")
            assert hasattr(app.state, "schemes")

            # Conversational agents (intake, guidance) get the short fail-fast
            # timeout so a hung Sarvam call can't stall the caller for the full
            # eligibility tail. The heavy agents (eligibility, reviewer) keep
            # the client default (None) and ride the longer 45s ceiling.
            orchestrator = app.state.conversation_manager._orchestrator
            assert orchestrator._intake._llm_timeout == 12.0
            assert orchestrator._guidance._llm_timeout == 12.0
            assert orchestrator._eligibility._llm_timeout is None
            assert orchestrator._reviewer._llm_timeout is None
