"""FastAPI application factory with lifespan management."""

from __future__ import annotations

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
    from vaidya.pipeline.translator import Translator
    from vaidya.sarvam.client import SarvamClient
    from vaidya.schemes.registry import get_schemes
    from vaidya.session.manager import SessionManager

    settings = Settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Starting Vaidya v%s", __version__)

    # Initialize Sarvam client
    client = SarvamClient(settings.sarvam_api_key)

    # Initialize knowledge store and load schemes
    store = KnowledgeStore(settings.chromadb_path)
    scheme_count = load_schemes_into_store(store)
    logger.info("Loaded %d schemes into knowledge store", scheme_count)

    # Get scheme list for agents
    schemes = get_schemes()

    # Initialize session manager
    session = SessionManager(settings.redis_url, settings.session_ttl_seconds)

    # Initialize agents
    intake = IntakeAgent(client, settings.intake_model)
    eligibility = EligibilityAgent(client, settings.eligibility_model, schemes, store=store)
    reviewer = ReviewerAgent(client, settings.reviewer_model, schemes)
    guidance = GuidanceAgent(client, settings.guidance_model)
    convergence = ConvergenceChecker()

    # Initialize orchestrator
    orchestrator = Orchestrator(
        client=client,
        intake=intake,
        eligibility=eligibility,
        reviewer=reviewer,
        guidance=guidance,
        convergence=convergence,
        fallback_model=settings.orchestrator_model,
        agent_timeout=settings.agent_timeout_seconds,
    )

    # Initialize pipeline
    translator = Translator(client)
    audit = AuditTrail()
    consent_tracker = ConsentTracker()
    conversation_manager = ConversationManager(
        orchestrator=orchestrator,
        session_manager=session,
        translator=translator,
        pii_masker=mask_pii,
        audit_trail=audit,
        consent_tracker=consent_tracker,
    )

    # Store in app state for dependency injection
    app.state.settings = settings
    app.state.client = client
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

    # Cleanup
    await session.close()
    logger.info("Vaidya shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Vaidya",
        description="Voice-first multi-agent healthcare scheme navigator for India",
        version=__version__,
        lifespan=lifespan,
    )

    # Register routes
    from vaidya.api.routes.compliance import router as compliance_router
    from vaidya.api.routes.conversation import router as conversation_router
    from vaidya.api.routes.health import router as health_router
    from vaidya.api.routes.schemes import router as schemes_router
    from vaidya.api.routes.simulate import router as simulate_router

    app.include_router(health_router, tags=["health"])
    app.include_router(conversation_router, prefix="/conversation", tags=["conversation"])
    app.include_router(simulate_router, prefix="/simulate", tags=["simulate"])
    app.include_router(schemes_router, prefix="/schemes", tags=["schemes"])
    app.include_router(compliance_router, prefix="/compliance", tags=["compliance"])

    return app
