"""Text-based simulation endpoint — primary demo interface."""

from __future__ import annotations

import hashlib
import time

from fastapi import APIRouter, Depends

from vaidya.dependencies import get_conversation_manager
from vaidya.models.api import SimulateRequest, SimulateResponse
from vaidya.pipeline.conversation import ConversationManager

router = APIRouter()


@router.post("/text", response_model=SimulateResponse)
async def simulate_text_conversation(
    request: SimulateRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
) -> SimulateResponse:
    """Simulate a full multi-turn conversation via text.

    This is the primary demo endpoint. Send a list of user utterances
    and receive the full conversation flow including eligibility results.
    """
    # Generate a deterministic call_id from the request
    phone_hash = hashlib.sha256(f"simulate-{time.time()}".encode()).hexdigest()[:16]

    call_id, welcome = await manager.start_conversation(
        phone_hash=phone_hash,
        language=request.language,
    )

    conversation: list[dict[str, str]] = [
        {"role": "assistant", "text": welcome},
    ]

    for user_text in request.turns:
        response = await manager.handle_turn(
            call_id=call_id,
            user_text=user_text,
            language=request.language,
        )
        conversation.append({"role": "user", "text": user_text})
        conversation.append({"role": "assistant", "text": response})

    # Get final state
    context = await manager.get_context(call_id)
    final_phase = context.phase.value if context else "unknown"
    eligible_schemes: list[str] = []
    if context and context.convergence_result:
        eligible_schemes = [m.scheme_name for m in context.convergence_result.all_eligible]

    return SimulateResponse(
        conversation=conversation,
        final_phase=final_phase,
        eligible_schemes=eligible_schemes,
    )
