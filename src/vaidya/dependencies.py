"""FastAPI dependency injection providers."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from vaidya.compliance.audit import AuditTrail
from vaidya.compliance.consent import ConsentTracker
from vaidya.config import Settings
from vaidya.knowledge.store import KnowledgeStore
from vaidya.models.scheme import SchemeRecord
from vaidya.pipeline.conversation import ConversationManager
from vaidya.sarvam.client import SarvamClient
from vaidya.session.manager import SessionManager


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_client(request: Request) -> SarvamClient:
    return cast(SarvamClient, request.app.state.client)


def get_store(request: Request) -> KnowledgeStore:
    return cast(KnowledgeStore, request.app.state.store)


def get_session(request: Request) -> SessionManager:
    return cast(SessionManager, request.app.state.session)


def get_conversation_manager(request: Request) -> ConversationManager:
    return cast(ConversationManager, request.app.state.conversation_manager)


def get_schemes(request: Request) -> list[SchemeRecord]:
    return cast(list[SchemeRecord], request.app.state.schemes)


def get_audit_trail(request: Request) -> AuditTrail:
    return cast(AuditTrail, request.app.state.audit_trail)


def get_consent_tracker(request: Request) -> ConsentTracker:
    return cast(ConsentTracker, request.app.state.consent_tracker)
