"""Eligibility Agent: structured field-by-field scheme matching via LLM."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from vaidya.agents.base import BaseAgent
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import (
    EligibilityResult,
    EligibilityVerdict,
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

# Fallback model used when the primary model call fails
_DEFAULT_FALLBACK_MODEL = SARVAM_30B

# Maximum number of schemes to send in a single LLM call to stay within
# context limits and keep latency reasonable
_MAX_SCHEMES_PER_CALL = 20

# Default number of results to fetch from vector search
_RAG_TOP_K = 10


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
    ) -> None:
        super().__init__(client=client, model=model, agent_name="eligibility")
        self._schemes = schemes
        self._store = store
        self._fallback_model = fallback_model

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Run eligibility evaluation against the scheme corpus.

        The *user_input* argument is unused here (the orchestrator passes
        an empty string); all data comes from context.user_profile.
        """
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
            # Retry with the fallback model
            model_used = self._fallback_model
            try:
                result = await self._evaluate(context, model_used)
            except Exception as fallback_exc:
                logger.error(
                    "Eligibility evaluation failed on both models",
                    extra={
                        "error": str(fallback_exc),
                        "call_id": context.call_id,
                    },
                    exc_info=True,
                )
                return self._fallback_response(context.language)

        elapsed_ms = (time.perf_counter() - start_ms) * 1000
        result.processing_time_ms = round(elapsed_ms, 1)
        result.model_used = model_used

        return AgentResponse(
            text="",  # Eligibility agent produces data, not spoken text
            eligibility_result=result,
            metadata={
                "schemes_evaluated": result.schemes_evaluated,
                "model_used": model_used,
                "processing_time_ms": result.processing_time_ms,
            },
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def _evaluate(
        self,
        context: ConversationContext,
        model: str,
    ) -> EligibilityResult:
        """Build prompt, call LLM, parse result into EligibilityResult.

        Retrieval strategy:
        1. If a KnowledgeStore is available, use vector search with state
           filtering plus forced inclusion of all central schemes.
        2. If retrieval returns nothing or the store is unavailable, fall
           back to the original ``_filter_schemes`` approach (serialize
           all schemes matching the user's state).
        """
        profile = context.user_profile

        # --- RAG retrieval path ---
        candidate_schemes: list[SchemeRecord] | None = None
        if self._store is not None:
            try:
                candidate_schemes = self._retrieve_schemes(context)
                logger.info(
                    "RAG retrieval returned %d candidate schemes",
                    len(candidate_schemes),
                    extra={"call_id": context.call_id},
                )
            except Exception as exc:
                logger.warning(
                    "RAG retrieval failed, falling back to full scheme list",
                    extra={"error": str(exc), "call_id": context.call_id},
                )
                candidate_schemes = None

        # --- Fallback: state-based filtering over the full list ---
        if not candidate_schemes:
            candidate_schemes = self._filter_schemes(profile.state)
            logger.info(
                "Using fallback state-filter: %d candidate schemes",
                len(candidate_schemes),
                extra={"call_id": context.call_id},
            )

        if not candidate_schemes:
            logger.info(
                "No candidate schemes after filtering",
                extra={"state": profile.state, "call_id": context.call_id},
            )
            return EligibilityResult(
                matches=[],
                processing_time_ms=0,
                model_used=model,
                schemes_evaluated=0,
            )

        profile_dict = profile.model_dump(mode="json", exclude_none=True)
        schemes_payload = self._serialize_schemes(candidate_schemes)

        system = prompts.render(
            "eligibility_system",
            user_profile=_json_str(profile_dict),
            schemes=_json_str(schemes_payload),
        )

        # Override the model on BaseAgent temporarily for this call
        original_model = self._model
        self._model = model
        try:
            raw = await self._call_llm_json(
                system,
                "Evaluate eligibility now.",
                reasoning_effort="high",
                max_tokens=4096,
                wiki_grounding=True,
            )
        finally:
            self._model = original_model

        return self._parse_result(raw, model, len(candidate_schemes))

    # ------------------------------------------------------------------
    # Scheme filtering
    # ------------------------------------------------------------------

    def _filter_schemes(self, user_state: str | None) -> list[SchemeRecord]:
        """Pre-filter schemes to those relevant to the user's state.

        Central schemes apply to all states. State schemes apply only
        if the state matches. When the user's state is unknown, all
        schemes are included (the LLM will mark state-specific ones
        as uncertain).
        """
        if not user_state:
            return self._schemes

        user_state_lower = user_state.lower().strip()
        filtered: list[SchemeRecord] = []

        for scheme in self._schemes:
            # Central schemes always apply
            if scheme.jurisdiction == Jurisdiction.CENTRAL:
                filtered.append(scheme)
                continue

            # State scheme: include if the state matches or if no
            # geographic restriction is specified
            if not scheme.geographic_restrictions:
                filtered.append(scheme)
                continue

            state_match = any(
                user_state_lower in r.lower() for r in scheme.geographic_restrictions
            )
            if state_match:
                filtered.append(scheme)

        return filtered

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def _retrieve_schemes(self, context: ConversationContext) -> list[SchemeRecord]:
        """Retrieve relevant schemes using hybrid RAG retrieval.

        Steps:
        1. Build a natural-language query from the user profile.
        2. Run state-filtered vector search via ChromaDB.
        3. Force-include all central schemes (they apply everywhere)
           so that embedding distance cannot accidentally exclude them.
        4. Deduplicate and return the merged list.
        """
        retrieval_start = time.perf_counter()

        query = self._build_retrieval_query(context)

        # State-filtered vector search
        state = context.user_profile.state
        state_code = state[:2].upper() if state else None

        candidates = self._store.search(query, n_results=_RAG_TOP_K, state_code=state_code)

        # Force-include ALL central schemes — they apply to every user
        central = [s for s in self._schemes if s.jurisdiction == Jurisdiction.CENTRAL]

        # Merge and deduplicate, preserving vector-search ordering first
        seen: set[str] = set()
        merged: list[SchemeRecord] = []
        for scheme in candidates + central:
            if scheme.scheme_id not in seen:
                seen.add(scheme.scheme_id)
                merged.append(scheme)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        logger.info(
            "Scheme retrieval completed",
            extra={
                "query": query,
                "state_code": state_code,
                "vector_hits": len(candidates),
                "central_added": len(central),
                "merged_total": len(merged),
                "retrieval_ms": round(retrieval_ms, 1),
                "call_id": context.call_id,
            },
        )

        return merged

    @staticmethod
    def _build_retrieval_query(context: ConversationContext) -> str:
        """Build a natural language query for scheme retrieval.

        Combines available profile fields into a descriptive sentence
        that ChromaDB's embedding function can match against
        ``description_for_embedding`` text stored for each scheme.
        """
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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_schemes(schemes: list[SchemeRecord]) -> list[dict[str, Any]]:
        """Convert scheme records to a compact dict list for the LLM prompt.

        We include only the fields the LLM needs for eligibility matching
        to keep the prompt within context limits.
        """
        out: list[dict[str, Any]] = []
        for s in schemes[:_MAX_SCHEMES_PER_CALL]:
            out.append(
                {
                    "scheme_id": s.scheme_id,
                    "canonical_name": s.canonical_name,
                    "jurisdiction": s.jurisdiction.value,
                    "state_code": s.state_code,
                    "income_thresholds": [t.model_dump(mode="json") for t in s.income_thresholds],
                    "secc_categories": s.secc_categories,
                    "occupation_included": s.occupation_included,
                    "occupation_excluded": s.occupation_excluded,
                    "exclusion_rules": [r.model_dump(mode="json") for r in s.exclusion_rules],
                    "age_criteria": (
                        s.age_criteria.model_dump(mode="json") if s.age_criteria else None
                    ),
                    "family_criteria": s.family_criteria.model_dump(mode="json"),
                    "geographic_restrictions": s.geographic_restrictions,
                    "coverage_amount_inr": s.coverage_amount_inr,
                    "coverage_type": s.coverage_type.value,
                    "covered_procedures": s.covered_procedures[:10],  # truncate for brevity
                }
            )
        return out

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

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
                verdict = _parse_verdict(verdict_str)
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_verdict(raw: str) -> EligibilityVerdict:
    """Map raw string to EligibilityVerdict, defaulting to UNCERTAIN."""
    mapping = {
        "eligible": EligibilityVerdict.ELIGIBLE,
        "ineligible": EligibilityVerdict.INELIGIBLE,
        "uncertain": EligibilityVerdict.UNCERTAIN,
    }
    return mapping.get(raw.strip().lower(), EligibilityVerdict.UNCERTAIN)


def _json_str(obj: Any) -> str:
    """Compact JSON string for prompt embedding."""
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
