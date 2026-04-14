"""Conversation state models for Vaidya voice sessions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from vaidya.models.scheme import (
    ConvergenceResult,
    EligibilityResult,
    GuidanceOutput,
    ReviewerResult,
)
from vaidya.models.user_profile import UserProfile


class ConversationPhase(StrEnum):
    """Finite-state phases of a Vaidya voice call."""

    WELCOME = "welcome"
    OPEN_ELICITATION = "open_elicitation"
    INTAKE = "intake"
    PROCESSING = "processing"
    RESULTS = "results"
    GUIDANCE = "guidance"
    CLOSURE = "closure"


class Turn(BaseModel):
    """A single conversational turn (user or assistant)."""

    model_config = ConfigDict(populate_by_name=True)

    role: str  # "user" | "assistant"
    text: str  # PII-masked text
    raw_text: str  # original pre-masking text
    language: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stt_confidence: float | None = None
    metadata: dict = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """Full mutable state for one phone call session.

    This is the object that moves through the pipeline: the orchestrator
    reads it, agents mutate it, and the session store persists it.
    """

    model_config = ConfigDict(populate_by_name=True)

    call_id: str
    phone_number_hash: str
    language: str
    phase: ConversationPhase
    transcript: list[Turn] = []

    # Progressive enrichment
    user_profile: UserProfile = Field(default_factory=UserProfile)
    eligibility_result: EligibilityResult | None = None
    reviewer_result: ReviewerResult | None = None
    convergence_result: ConvergenceResult | None = None
    guidance_output: GuidanceOutput | None = None

    # Intake bookkeeping
    intake_question_index: int = 0
    retry_count: int = 0
    emotional_distress_detected: bool = False

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        text: str,
        raw_text: str,
        language: str | None = None,
        stt_confidence: float | None = None,
        metadata: dict | None = None,
    ) -> Turn:
        """Append a turn to the transcript and bump `updated_at`."""
        turn = Turn(
            role=role,
            text=text,
            raw_text=raw_text,
            language=language or self.language,
            stt_confidence=stt_confidence,
            metadata=metadata or {},
        )
        self.transcript.append(turn)
        self.updated_at = datetime.now(UTC)
        return turn

    @property
    def full_transcript_text(self) -> str:
        """Join all turns as ``[role] text`` separated by newlines."""
        return "\n".join(f"[{t.role}] {t.text}" for t in self.transcript)
