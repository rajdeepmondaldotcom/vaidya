"""Vaidya domain models -- re-exports for convenient importing.

Usage::

    from vaidya.models import UserProfile, SchemeRecord, ConversationContext
"""

from __future__ import annotations

from vaidya.models.api import (
    AgentResponse,
    HealthResponse,
    SchemeResponse,
    SimulateRequest,
    SimulateResponse,
    StartConversationRequest,
    StartConversationResponse,
    TurnRequest,
    TurnResponse,
)
from vaidya.models.conversation import (
    ConversationContext,
    ConversationPhase,
    Turn,
)
from vaidya.models.scheme import (
    AgeCriteria,
    ConfidenceLevel,
    ConvergenceResult,
    DisagreementRecord,
    Document,
    EligibilityResult,
    EligibilityVerdict,
    ExclusionRule,
    ExclusionType,
    FamilyCriteria,
    GuidanceOutput,
    IncomeThreshold,
    Jurisdiction,
    ReviewerResult,
    SchemeCoverageType,
    SchemeMatch,
    SchemeRecord,
)
from vaidya.models.user_profile import (
    CoverageType,
    IncomeCategory,
    OccupationType,
    UserProfile,
)

__all__ = [
    # user_profile
    "CoverageType",
    "IncomeCategory",
    "OccupationType",
    "UserProfile",
    # scheme
    "AgeCriteria",
    "ConfidenceLevel",
    "ConvergenceResult",
    "DisagreementRecord",
    "Document",
    "EligibilityResult",
    "EligibilityVerdict",
    "ExclusionRule",
    "ExclusionType",
    "FamilyCriteria",
    "GuidanceOutput",
    "IncomeThreshold",
    "Jurisdiction",
    "ReviewerResult",
    "SchemeCoverageType",
    "SchemeMatch",
    "SchemeRecord",
    # conversation
    "ConversationContext",
    "ConversationPhase",
    "Turn",
    # api
    "AgentResponse",
    "HealthResponse",
    "SchemeResponse",
    "SimulateRequest",
    "SimulateResponse",
    "StartConversationRequest",
    "StartConversationResponse",
    "TurnRequest",
    "TurnResponse",
]
