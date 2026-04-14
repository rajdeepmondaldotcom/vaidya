"""FastAPI dependency injection providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from vaidya.compliance.audit import AuditTrail
    from vaidya.compliance.consent import ConsentTracker
    from vaidya.config import Settings
    from vaidya.knowledge.store import KnowledgeStore
    from vaidya.models.scheme import SchemeRecord
    from vaidya.pipeline.conversation import ConversationManager
    from vaidya.sarvam.client import SarvamClient
    from vaidya.session.manager import SessionManager


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_client(request: Request) -> SarvamClient:
    return request.app.state.client  # type: ignore[no-any-return]


def get_store(request: Request) -> KnowledgeStore:
    return request.app.state.store  # type: ignore[no-any-return]


def get_session(request: Request) -> SessionManager:
    return request.app.state.session  # type: ignore[no-any-return]


def get_conversation_manager(request: Request) -> ConversationManager:
    return request.app.state.conversation_manager  # type: ignore[no-any-return]


def get_schemes(request: Request) -> list[SchemeRecord]:
    return request.app.state.schemes  # type: ignore[no-any-return]


def get_audit_trail(request: Request) -> AuditTrail:
    return request.app.state.audit_trail  # type: ignore[no-any-return]


def get_consent_tracker(request: Request) -> ConsentTracker:
    return request.app.state.consent_tracker  # type: ignore[no-any-return]
