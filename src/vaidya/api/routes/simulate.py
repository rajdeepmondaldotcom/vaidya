"""Text-based simulation endpoint -- primary demo interface."""

from __future__ import annotations

import hashlib
import time

from fastapi import APIRouter, Depends

from vaidya.config import Settings
from vaidya.dependencies import get_client, get_conversation_manager, get_settings
from vaidya.models.api import SimulateRequest, SimulateResponse
from vaidya.pipeline.conversation import ConversationManager
from vaidya.sarvam.client import SarvamClient

router = APIRouter()


async def _run_simulation_turns(
    manager: ConversationManager,
    call_id: str,
    turns: list[str],
    language: str,
) -> list[dict[str, str]]:
    """Execute each user turn and collect the conversation transcript."""
    conversation: list[dict[str, str]] = []
    for user_text in turns:
        response = await manager.handle_turn(
            call_id=call_id,
            user_text=user_text,
            language=language,
        )
        conversation.append({"role": "user", "text": user_text})
        conversation.append({"role": "assistant", "text": response})
    return conversation


def _build_simulation_response(
    conversation: list[dict[str, str]],
    context: object | None,
    client: SarvamClient | None,
    call_id: str,
) -> SimulateResponse:
    """Build the SimulateResponse from final context state."""
    final_phase = context.phase.value if context else "unknown"  # type: ignore[union-attr]
    eligible_scheme_ids: list[str] = []
    eligible_scheme_names: list[str] = []
    if context and getattr(context, "convergence_result", None):
        eligible_scheme_ids = [m.scheme_id for m in context.convergence_result.all_eligible]  # type: ignore[union-attr]
        eligible_scheme_names = [m.scheme_name for m in context.convergence_result.all_eligible]  # type: ignore[union-attr]

    session_cost = context.metadata.get("session_cost_inr", 0.0) if context else 0.0  # type: ignore[union-attr]
    total_cost = round(client.costs.cost_for_call(call_id), 4) if client else None

    return SimulateResponse(
        conversation=conversation,
        final_phase=final_phase,
        eligible_schemes=eligible_scheme_ids,
        eligible_scheme_names=eligible_scheme_names,
        session_cost_inr=session_cost,
        total_cost_inr=total_cost,
    )


@router.post("/text", response_model=SimulateResponse)
async def simulate_text_conversation(
    request: SimulateRequest,
    manager: ConversationManager = Depends(get_conversation_manager),
    client: SarvamClient = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> SimulateResponse:
    """Simulate a full multi-turn conversation via text."""
    phone_hash = hashlib.sha256(f"simulate-{time.time()}".encode()).hexdigest()[:16]
    turns = request.turns[: settings.max_simulation_turns]

    call_id, welcome = await manager.start_conversation(
        phone_hash=phone_hash,
        language=request.language,
    )

    conversation: list[dict[str, str]] = [{"role": "assistant", "text": welcome}]
    conversation.extend(await _run_simulation_turns(manager, call_id, turns, request.language))

    context = await manager.get_context(call_id)
    return _build_simulation_response(conversation, context, client, call_id)
