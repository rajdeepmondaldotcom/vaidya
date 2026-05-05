"""Conversation management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from vaidya.dependencies import get_client, get_conversation_manager
from vaidya.models.api import (
    StartConversationRequest,
    StartConversationResponse,
    TurnRequest,
    TurnResponse,
)
from vaidya.pipeline.conversation import ConversationManager
from vaidya.sarvam.client import SarvamClient
from vaidya.validation import ValidationError, validate_language

router = APIRouter()


@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(
    request: StartConversationRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
) -> StartConversationResponse:
    """Start a new conversation session."""
    try:
        validated_lang = validate_language(request.language, channel=request.channel)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None

    call_id, welcome = await manager.start_conversation(
        phone_hash=request.phone_number_hash,
        language=validated_lang,
        channel=request.channel,
    )
    return StartConversationResponse(
        call_id=call_id,
        phase="welcome",
        message=welcome,
    )


async def _build_turn_response(
    call_id: str,
    response_text: str,
    manager: ConversationManager,
    client: SarvamClient | None,
) -> TurnResponse:
    """Reload context after a turn and build the response with metadata."""
    updated = await manager.get_context(call_id)
    phase = updated.phase.value if updated else "unknown"
    schemes_found = None
    if updated and updated.convergence_result:
        schemes_found = len(updated.convergence_result.all_eligible)

    cost_so_far = round(client.costs.cost_for_call(call_id), 4) if client else None
    return TurnResponse(
        text=response_text,
        phase=phase,
        schemes_found=schemes_found,
        cost_so_far_inr=cost_so_far,
    )


@router.post("/{call_id}/turn", response_model=TurnResponse)
async def conversation_turn(
    call_id: str,
    request_body: TurnRequest,
    request: Request,
    manager: ConversationManager = Depends(get_conversation_manager),
    client: SarvamClient = Depends(get_client),
) -> TurnResponse:
    """Send a user message and get the assistant response."""
    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="Client disconnected")

    response_text = await manager.handle_turn(
        call_id=call_id,
        user_text=request_body.text,
        language=request_body.language,
        stt_confidence=request_body.stt_confidence,
    )
    return await _build_turn_response(call_id, response_text, manager, client)


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
