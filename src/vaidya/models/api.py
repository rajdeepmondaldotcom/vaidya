"""Request / response models for the Vaidya HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vaidya.models.conversation import ConversationPhase
from vaidya.models.scheme import (
    ConvergenceResult,
    EligibilityResult,
    GuidanceOutput,
    ReviewerResult,
)
from vaidya.models.user_profile import UserProfile

# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------


class StartConversationRequest(BaseModel):
    """POST /conversation/start"""

    phone_number_hash: str
    language: str = "hi-IN"
    channel: str = "voice"  # voice | whatsapp | sms | web


class StartConversationResponse(BaseModel):
    """Response to POST /conversation/start"""

    call_id: str
    phase: str
    message: str


class TurnRequest(BaseModel):
    """POST /conversation/{call_id}/turn"""

    text: str
    language: str | None = None
    stt_confidence: float = 1.0
    channel: str = "text"  # voice | whatsapp | sms | web | text


class TurnResponse(BaseModel):
    """Response to a single conversational turn."""

    text: str
    phase: str
    schemes_found: int | None = None
    cost_so_far_inr: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    """POST /simulate -- replay a full conversation for testing."""

    turns: list[str]
    language: str = "hi-IN"


class SimulateResponse(BaseModel):
    """Response to a simulation run."""

    conversation: list[dict[str, str]]
    final_phase: str
    eligible_schemes: list[str]  # scheme IDs for eval scoring
    eligible_scheme_names: list[str] = Field(default_factory=list)
    session_cost_inr: float = 0.0
    total_cost_inr: float | None = None


# ---------------------------------------------------------------------------
# Scheme catalog
# ---------------------------------------------------------------------------


class SchemeResponse(BaseModel):
    """GET /schemes/{scheme_id}"""

    scheme_id: str
    canonical_name: str
    coverage_amount_inr: int
    jurisdiction: str
    state_code: str | None = None
    description: str


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /health"""

    status: str
    version: str


# ---------------------------------------------------------------------------
# Agent internal response
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Unified response envelope returned by any pipeline agent."""

    text: str
    phase_transition: ConversationPhase | None = None
    updated_profile: UserProfile | None = None
    eligibility_result: EligibilityResult | None = None
    reviewer_result: ReviewerResult | None = None
    convergence_result: ConvergenceResult | None = None
    guidance_output: GuidanceOutput | None = None
    already_localized: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
