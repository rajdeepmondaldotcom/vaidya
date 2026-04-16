"""Eligibility Agent: structured field-by-field scheme matching via LLM."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from vaidya.agents.base import BaseAgent
from vaidya.agents.constants import RAG_TOP_K
from vaidya.agents.scheme_utils import (
    filter_schemes_by_state,
    json_compact,
    parse_verdict,
    serialize_for_prompt,
)
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import (
    EligibilityResult,
    Jurisdiction,
    SchemeMatch,
    SchemeRecord,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient
from vaidya.sarvam.models import SARVAM_30B

if TYPE_CHECKING:
    from vaidya.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

_DEFAULT_FALLBACK_MODEL = SARVAM_30B


class EligibilityAgent(BaseAgent):
    """Determines scheme eligibility using structured LLM evaluation.

    When a :class:`KnowledgeStore` is provided, the agent performs hybrid
    RAG retrieval (vector search with state filtering + central-scheme
    inclusion) before sending candidates to the LLM.  When the store is
    unavailable or returns no results, the agent falls back to the
    original approach of serializing all schemes.

    For each candidate scheme, the LLM performs field-by-field matching
    against the user profile, checking income thresholds, occupation,
    geographic restrictions, exclusion rules, and family criteria.
    """

    def __init__(
        self,
        client: SarvamClient,
        model: str,
        schemes: list[SchemeRecord],
        store: KnowledgeStore | None = None,
        fallback_model: str = _DEFAULT_FALLBACK_MODEL,
        reasoning_effort: str = "high",
    ) -> None:
        super().__init__(client=client, model=model, agent_name="eligibility")
        self._schemes = schemes
        self._store = store
        self._fallback_model = fallback_model
        self._reasoning_effort = reasoning_effort

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Run eligibility evaluation against the scheme corpus."""
        start_ms = time.perf_counter()
        model_used = self._model

        try:
            result = await self._evaluate(context, model_used)
        except Exception as primary_exc:
            logger.warning(
                "Primary model failed, falling back",
                extra={
                    "primary_model": model_used,
                    "fallback_model": self._fallback_model,
                    "error": str(primary_exc),
                    "call_id": context.call_id,
                },
            )
            model_used = self._fallback_model
            try:
                result = await self._evaluate(context, model_used)
            except Exception as fallback_exc:
                logger.error(
                    "Eligibility evaluation failed on both models",
                    extra={"error": str(fallback_exc), "call_id": context.call_id},
                    exc_info=True,
                )
                return self._fallback_response(context.language)

        elapsed_ms = (time.perf_counter() - start_ms) * 1000
        result.processing_time_ms = round(elapsed_ms, 1)
        result.model_used = model_used

        return AgentResponse(
            text="",
            eligibility_result=result,
            metadata={
                "schemes_evaluated": result.schemes_evaluated,
                "model_used": model_used,
                "processing_time_ms": result.processing_time_ms,
            },
        )

    def _get_candidate_schemes(
        self,
        context: ConversationContext,
    ) -> list[SchemeRecord]:
        """Get candidate schemes via RAG retrieval with state-filter fallback."""
        if self._store is not None:
            try:
                candidates = self._retrieve_schemes(context)
                logger.info(
                    "RAG retrieval returned %d candidate schemes",
                    len(candidates),
                    extra={"call_id": context.call_id},
                )
                if candidates:
                    return candidates
            except Exception as exc:
                logger.warning(
                    "RAG retrieval failed, falling back to full scheme list",
                    extra={"error": str(exc), "call_id": context.call_id},
                )

        candidates = self._filter_schemes(context.user_profile.state)
        logger.info(
            "Using fallback state-filter: %d candidate schemes",
            len(candidates),
            extra={"call_id": context.call_id},
        )
        return candidates

    async def _evaluate(
        self,
        context: ConversationContext,
        model: str,
    ) -> EligibilityResult:
        """Get candidates, build prompt, call LLM, and parse the result."""
        candidate_schemes = self._get_candidate_schemes(context)

        if not candidate_schemes:
            logger.info(
                "No candidate schemes after filtering",
                extra={"state": context.user_profile.state, "call_id": context.call_id},
            )
            return EligibilityResult(
                matches=[],
                processing_time_ms=0,
                model_used=model,
                schemes_evaluated=0,
            )

        profile_dict = context.user_profile.model_dump(mode="json", exclude_none=True)
        schemes_payload = self._serialize_schemes(candidate_schemes)

        system = prompts.render(
            "eligibility_system",
            user_profile=json_compact(profile_dict),
            schemes=json_compact(schemes_payload),
        )

        raw = await self._call_llm_json(
            system,
            "Evaluate eligibility now.",
            model=model,
            reasoning_effort=self._reasoning_effort,
            max_tokens=4096,
            wiki_grounding=True,
        )
        return self._parse_result(raw, model, len(candidate_schemes))

    def _filter_schemes(self, user_state: str | None) -> list[SchemeRecord]:
        """Pre-filter schemes to those relevant to the user's state."""
        return filter_schemes_by_state(self._schemes, user_state)

    def _resolve_and_merge(
        self,
        candidates: list[SchemeRecord],
    ) -> list[SchemeRecord]:
        """Resolve vector hits against the registry and merge with central schemes."""
        registry_map = {s.scheme_id: s for s in self._schemes}

        resolved = [registry_map.get(stub.scheme_id, stub) for stub in candidates]
        central = [s for s in self._schemes if s.jurisdiction == Jurisdiction.CENTRAL]

        seen: set[str] = set()
        merged: list[SchemeRecord] = []
        for scheme in resolved + central:
            if scheme.scheme_id not in seen:
                seen.add(scheme.scheme_id)
                merged.append(scheme)
        return merged

    def _retrieve_schemes(self, context: ConversationContext) -> list[SchemeRecord]:
        """Retrieve relevant schemes via hybrid RAG retrieval.

        Combines state-filtered vector search with forced central-scheme
        inclusion, resolving stubs against the authoritative registry.
        """
        from vaidya.utils.states import state_name_to_code

        retrieval_start = time.perf_counter()
        query = self._build_retrieval_query(context)

        state = context.user_profile.state
        state_code = state_name_to_code(state) if state else None
        candidates = self._store.search(query, n_results=RAG_TOP_K, state_code=state_code)  # type: ignore[union-attr]

        merged = self._resolve_and_merge(candidates)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        logger.info(
            "Scheme retrieval completed",
            extra={
                "query_length": len(query),
                "state_code": state_code,
                "vector_hits": len(candidates),
                "merged_total": len(merged),
                "retrieval_ms": round(retrieval_ms, 1),
                "call_id": context.call_id,
            },
        )
        return merged

    @staticmethod
    def _build_retrieval_query(context: ConversationContext) -> str:
        """Build a natural language query from the user profile for vector search."""
        profile = context.user_profile
        parts: list[str] = []

        if profile.state:
            parts.append(f"healthcare scheme in {profile.state}")
        if profile.health_need:
            parts.append(f"for {profile.health_need}")
        if profile.occupation_type.value != "unknown":
            parts.append(f"for {profile.occupation_type.value} worker")
        if profile.income_bracket.value != "unknown":
            parts.append(f"income {profile.income_bracket.value}")

        return " ".join(parts) if parts else "government healthcare scheme India"

    @staticmethod
    def _serialize_schemes(schemes: list[SchemeRecord]) -> list[dict[str, Any]]:
        """Convert scheme records to a compact dict list for the LLM prompt."""
        return serialize_for_prompt(schemes, include_procedures=True)

    def _parse_result(
        self,
        raw: dict[str, Any],
        model_used: str,
        schemes_evaluated: int,
    ) -> EligibilityResult:
        """Parse LLM JSON into a validated EligibilityResult."""
        if isinstance(raw, list):
            raw = {"matches": raw}
        if not isinstance(raw, dict) or raw.get("_parse_error"):
            logger.warning("LLM returned unparseable JSON for eligibility")
            return EligibilityResult(
                matches=[],
                processing_time_ms=0,
                model_used=model_used,
                schemes_evaluated=schemes_evaluated,
            )

        matches: list[SchemeMatch] = []
        for item in raw.get("matches", []):
            try:
                verdict_str = str(item.get("verdict", "uncertain")).lower()
                verdict = parse_verdict(verdict_str)
                match = SchemeMatch(
                    scheme_id=str(item.get("scheme_id", "")),
                    scheme_name=str(item.get("scheme_name", item.get("scheme_id", ""))),
                    verdict=verdict,
                    confidence=float(item.get("confidence", 0.0)),
                    reasoning_trace=str(item.get("reasoning_trace", "")),
                    matched_criteria=item.get("matched_criteria", []),
                    failed_criteria=item.get("failed_criteria", []),
                    coverage_summary=str(item.get("coverage_summary", "")),
                )
                matches.append(match)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed scheme match",
                    extra={"error": str(exc), "item": str(item)[:200]},
                )

        return EligibilityResult(
            matches=matches,
            processing_time_ms=0,  # filled by caller
            model_used=model_used,
            schemes_evaluated=raw.get("schemes_evaluated", schemes_evaluated),
        )
