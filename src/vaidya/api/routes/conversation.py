"""Conversation management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from vaidya.dependencies import get_conversation_manager
from vaidya.models.api import (
    StartConversationRequest,
    StartConversationResponse,
    TurnRequest,
    TurnResponse,
)
from vaidya.pipeline.conversation import ConversationManager

router = APIRouter()


@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(
    request: StartConversationRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
) -> StartConversationResponse:
    """Start a new conversation session."""
    call_id, welcome = await manager.start_conversation(
        phone_hash=request.phone_number_hash,
        language=request.language,
    )
    return StartConversationResponse(
        call_id=call_id,
        phase="welcome",
        message=welcome,
    )


@router.post("/{call_id}/turn", response_model=TurnResponse)
async def conversation_turn(
    call_id: str,
    request: TurnRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
) -> TurnResponse:
    """Send a user message and get the assistant response."""
    context = await manager.get_context(call_id)
    if context is None:
        raise HTTPException(status_code=404, detail=f"Conversation {call_id} not found")

    response_text = await manager.handle_turn(
        call_id=call_id,
        user_text=request.text,
        language=request.language,
        stt_confidence=request.stt_confidence,
    )

    # Reload context after turn
    updated = await manager.get_context(call_id)
    phase = updated.phase.value if updated else "unknown"
    schemes_found = None
    if updated and updated.convergence_result:
        schemes_found = len(updated.convergence_result.all_eligible)

    return TurnResponse(
        text=response_text,
        phase=phase,
        schemes_found=schemes_found,
    )


@router.get("/{call_id}")
async def get_conversation(
    call_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),
) -> dict:
    """Get current conversation state."""
    context = await manager.get_context(call_id)
    if context is None:
        raise HTTPException(status_code=404, detail=f"Conversation {call_id} not found")

    return {
        "call_id": context.call_id,
        "phase": context.phase.value,
        "language": context.language,
        "user_profile": context.user_profile.model_dump(),
        "transcript": [
            {"role": t.role, "text": t.text, "language": t.language} for t in context.transcript
        ],
        "eligible_schemes": (
            [m.scheme_name for m in context.convergence_result.all_eligible]
            if context.convergence_result
            else None
        ),
        "intake_progress": f"{context.intake_question_index}/5",
    }
