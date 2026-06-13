"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vaidya import __version__

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize services on startup, clean up on shutdown."""
    from vaidya.agents.convergence import ConvergenceChecker
    from vaidya.agents.eligibility import EligibilityAgent
    from vaidya.agents.guidance import GuidanceAgent
    from vaidya.agents.intake import IntakeAgent
    from vaidya.agents.orchestrator import Orchestrator
    from vaidya.agents.reviewer import ReviewerAgent
    from vaidya.compliance.audit import AuditTrail
    from vaidya.compliance.consent import ConsentTracker
    from vaidya.compliance.pii import mask_pii
    from vaidya.config import Settings
    from vaidya.knowledge.loader import load_schemes_into_store
    from vaidya.knowledge.store import KnowledgeStore
    from vaidya.pipeline.conversation import ConversationManager
    from vaidya.pipeline.degradation import DegradationManager
    from vaidya.pipeline.translator import Translator
    from vaidya.sarvam.client import SarvamClient
    from vaidya.sarvam.tts_cache import TTSCache
    from vaidya.schemes.registry import get_schemes
    from vaidya.session.manager import SessionManager

    settings = Settings()

    # Bug 5: Validate Sarvam API key at startup
    if not settings.sarvam_api_key:
        raise RuntimeError(
            "SARVAM_API_KEY environment variable is required. "
            "Get one free at https://dashboard.sarvam.ai"
        )

    # Bug 11: Optional Sentry error tracking
    if settings.sentry_dsn:
        try:
            import sentry_sdk  # type: ignore[import-not-found]

            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.environment,
                traces_sample_rate=0.1 if settings.environment == "production" else 1.0,
            )
            logger.info("Sentry error tracking enabled")
        except ImportError:
            logger.warning("sentry-sdk not installed, error tracking disabled")

    # Configure structured logging
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
                if settings.environment == "development"
                else structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, settings.log_level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        pass  # structlog is optional; fall back to stdlib
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Starting Vaidya v%s", __version__)

    # Initialize Sarvam client
    client = SarvamClient(settings.sarvam_api_key, timeout=settings.llm_timeout_seconds)

    # Cache synthesized audio for fixed/templated prompts (greeting, the five
    # intake questions, processing filler, silence nudges, closure) so they
    # aren't re-synthesized on every call. Wraps the same client, so cache
    # misses still flow through its circuit breaker and cost tracking. Attach
    # it to the client so the voice pipeline's agent processor — built without
    # a cache kwarg — can reach this one shared instance via its manager.
    tts_cache = TTSCache(client)
    # setattr (not direct assignment) keeps this out of SarvamClient's declared
    # surface: the cache is an app-level singleton bolted onto the shared client
    # so the voice pipeline can reach it; the client itself stays cache-agnostic.
    client.tts_cache = tts_cache  # type: ignore[attr-defined]

    # Warm the Sarvam TLS connection up front so the first real STT/LLM/TTS
    # call on a fresh deploy isn't cold (DNS + TLS handshake moved off the
    # caller's first request). Best-effort and bounded — prewarm() swallows
    # its own failures and never raises; skipped under the test environment so
    # the lifespan smoke test stays hermetic (no outbound network).
    if settings.environment != "test":
        await client.prewarm()

    # Initialize knowledge store and load schemes
    store = KnowledgeStore(
        chromadb_path=settings.chromadb_path,
        chromadb_host=settings.chromadb_host,
        chromadb_port=settings.chromadb_port,
    )
    scheme_count = load_schemes_into_store(store)
    logger.info("Loaded %d schemes into knowledge store", scheme_count)

    # Bug 8: Health-check ChromaDB (warn, don't crash)
    if not store.is_healthy():
        logger.warning("ChromaDB health check failed - scheme search may be degraded")

    # Get scheme list for agents
    schemes = get_schemes()

    # Initialize session manager
    session = SessionManager(
        settings.redis_url, settings.session_ttl_seconds, settings.redis_max_connections
    )

    # Initialize agents
    intake = IntakeAgent(
        client,
        settings.intake_model,
        reasoning_effort=settings.intake_reasoning_effort,
    )
    eligibility = EligibilityAgent(
        client,
        settings.eligibility_model,
        schemes,
        store=store,
        reasoning_effort=settings.eligibility_reasoning_effort,
        batch_size=settings.scheme_eval_batch_size,
        max_parallel_batches=settings.scheme_eval_max_parallel_batches,
        retrieval_rank_top_k=settings.scheme_retrieval_rank_top_k,
    )
    reviewer = ReviewerAgent(
        client,
        settings.reviewer_model,
        schemes,
        reasoning_effort=settings.reviewer_reasoning_effort,
        batch_size=settings.scheme_eval_batch_size,
        max_parallel_batches=settings.scheme_eval_max_parallel_batches,
    )
    guidance = GuidanceAgent(
        client,
        settings.guidance_model,
        reasoning_effort=settings.guidance_reasoning_effort,
    )
    convergence = ConvergenceChecker()

    # Fail-fast timeout for the CONVERSATIONAL agents only (intake, guidance):
    # their fast 30b calls normally finish in ~2s, so a hung Sarvam call must
    # not stall the caller for the full eligibility tail before retrying. Set
    # post-construction via the BaseAgent attribute (mirrors the client.tts_cache
    # pattern above) so subclass __init__ signatures stay untouched. Eligibility
    # and reviewer keep the client default (llm_timeout_seconds) for their heavy
    # 105b batches.
    intake._llm_timeout = settings.conversational_llm_timeout_seconds
    guidance._llm_timeout = settings.conversational_llm_timeout_seconds
    # Opt-in heuristic intake fast path (off by default — LLM-first showcases
    # sarvam-30b's multilingual extraction; enable for cost-sensitive scale).
    intake._fast_path_enabled = settings.intake_fast_path_enabled

    # Initialize orchestrator
    audit = AuditTrail()
    consent_tracker = ConsentTracker()
    degradation_manager = DegradationManager()
    orchestrator = Orchestrator(
        client=client,
        intake=intake,
        eligibility=eligibility,
        reviewer=reviewer,
        guidance=guidance,
        convergence=convergence,
        fallback_model=settings.orchestrator_model,
        agent_timeout=settings.agent_timeout_seconds,
        consent_tracker=consent_tracker,
        degradation=degradation_manager,
    )

    # Initialize pipeline
    translator = Translator(client)
    conversation_manager = ConversationManager(
        orchestrator=orchestrator,
        session_manager=session,
        translator=translator,
        audit_trail=audit,
        consent_tracker=consent_tracker,
        pii_masker=mask_pii,
        sarvam_client=client,
    )

    # Store in app state for dependency injection
    app.state.settings = settings
    app.state.client = client
    app.state.tts_cache = tts_cache
    app.state.store = store
    app.state.session = session
    app.state.conversation_manager = conversation_manager
    app.state.schemes = schemes
    app.state.audit_trail = audit
    app.state.consent_tracker = consent_tracker

    logger.info(
        "Vaidya ready — %d schemes, serving on %s:%d", scheme_count, settings.host, settings.port
    )

    yield

    # Graceful shutdown
    logger.info("Shutting down Vaidya...")
    try:
        await asyncio.wait_for(session.close(), timeout=10.0)
    except TimeoutError:
        logger.warning("Redis close timed out during shutdown")
    except Exception as exc:
        logger.error("Error during shutdown: %s", exc)
    logger.info("Vaidya shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from fastapi.middleware.cors import CORSMiddleware

    from vaidya.api.middleware import RateLimitMiddleware, RequestIdMiddleware

    app = FastAPI(
        title="Vaidya",
        description="Voice-first multi-agent healthcare scheme navigator for India",
        version=__version__,
        lifespan=lifespan,
    )

    # Middleware (outermost first)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # CORS: In dev (no origins configured), allow * without credentials.
    # In prod (origins set), allow only those with credentials.
    from vaidya.config import Settings as _CORSSettings

    _cors_settings = _CORSSettings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_settings.allowed_origins if _cors_settings.allowed_origins else ["*"],
        allow_credentials=bool(_cors_settings.allowed_origins),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    # Register routes
    from vaidya.api.routes.compliance import router as compliance_router
    from vaidya.api.routes.conversation import router as conversation_router
    from vaidya.api.routes.health import router as health_router
    from vaidya.api.routes.schemes import router as schemes_router
    from vaidya.api.routes.simulate import router as simulate_router
    from vaidya.api.routes.voice import router as voice_router

    app.include_router(health_router, tags=["health"])
    app.include_router(conversation_router, prefix="/conversation", tags=["conversation"])
    app.include_router(simulate_router, prefix="/simulate", tags=["simulate"])
    app.include_router(schemes_router, prefix="/schemes", tags=["schemes"])
    app.include_router(compliance_router, prefix="/compliance", tags=["compliance"])
    app.include_router(voice_router, prefix="/voice", tags=["voice"])

    return app
