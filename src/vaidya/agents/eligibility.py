"""Eligibility Agent: structured field-by-field scheme matching via LLM."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from vaidya.agents.base import BaseAgent
from vaidya.agents.constants import (
    MAX_PARALLEL_SCHEME_BATCHES,
    MAX_SCHEMES_PER_LLM_CALL,
    RAG_TOP_K,
)
from vaidya.agents.scheme_utils import (
    batch_schemes,
    filter_schemes_by_state,
    json_compact,
    missing_candidate_ids,
    normalize_matches_for_candidates,
    parse_verdict,
    serialize_for_prompt,
    uncertain_matches_for_missing,
)
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import (
    EligibilityResult,
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

# Placeholder substituted into the rendered system prompt once per call so the
# invariant prompt body (instructions + user profile) is rendered a single time
# and each batch only swaps in its own serialized scheme payload.
_SCHEMES_PLACEHOLDER = "\x00__VAIDYA_SCHEMES_PAYLOAD__\x00"

# Keys used to cache the RAG retrieval ordering across turns of one session.
_RAG_CACHE_FINGERPRINT_KEY = "eligibility_rag_fingerprint"
_RAG_CACHE_ORDER_KEY = "eligibility_rag_scheme_id_order"


class EligibilityAgent(BaseAgent):
    """Determines scheme eligibility using structured LLM evaluation.

    When a :class:`KnowledgeStore` is provided, the agent uses vector
    retrieval to rank and prune the applicable candidate set to the
    most-relevant top-k. This keeps the LLM workload (and latency) flat as the
    corpus grows. The retrieval ranking is cached per session against a profile
    fingerprint, so unchanged profiles reuse the prior ordering without
    re-querying the store. When NO store is available, the full applicable set
    is evaluated -- retrieval-based pruning is the only mechanism that may drop
    a state-applicable scheme, never a hard cap.

    For each candidate scheme, the LLM performs field-by-field matching
    against the user profile, checking income thresholds, occupation,
    geographic restrictions, exclusion rules, and family criteria. Large
    candidate sets are evaluated in bounded, concurrent LLM batches that all
    share a single rendered system prompt.
    """

    def __init__(
        self,
        client: SarvamClient,
        model: str,
        schemes: list[SchemeRecord],
        store: KnowledgeStore | None = None,
        fallback_model: str = _DEFAULT_FALLBACK_MODEL,
        reasoning_effort: str = "high",
        batch_size: int = MAX_SCHEMES_PER_LLM_CALL,
        max_parallel_batches: int = MAX_PARALLEL_SCHEME_BATCHES,
        retrieval_rank_top_k: int = RAG_TOP_K,
    ) -> None:
        super().__init__(client=client, model=model, agent_name="eligibility")
        self._schemes = schemes
        self._store = store
        self._fallback_model = fallback_model
        self._reasoning_effort = reasoning_effort
        self._batch_size = max(1, batch_size)
        self._max_parallel_batches = max(1, max_parallel_batches)
        self._retrieval_rank_top_k = max(1, retrieval_rank_top_k)

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
                "eligibility_batch_count": context.metadata.get("eligibility_batch_count", 0),
                "eligibility_missing_scheme_ids": context.metadata.get(
                    "eligibility_missing_scheme_ids",
                    [],
                ),
            },
        )

    def _get_candidate_schemes(
        self,
        context: ConversationContext,
    ) -> list[SchemeRecord]:
        """Get candidate schemes for evaluation.

        With a knowledge store, returns the RAG-ranked top-k of the
        state-applicable set (pruned, so the batched LLM workload stays flat as
        the corpus grows). Without a store -- or if retrieval yields no usable
        hits or errors -- returns the FULL applicable set so a weak/empty
        retrieval can never silently drop a genuinely applicable scheme.
        """
        applicable = self._filter_schemes(context.user_profile.state)
        if not applicable:
            return []

        if self._store is not None:
            try:
                candidates = self._rank_schemes_by_retrieval(context, applicable)
                logger.info(
                    "RAG ranked %d applicable candidate schemes",
                    len(candidates),
                    extra={"call_id": context.call_id},
                )
                if candidates:
                    return candidates
            except Exception as exc:
                logger.warning(
                    "RAG ranking failed, falling back to applicable scheme order",
                    extra={"error": str(exc), "call_id": context.call_id},
                )

        logger.info(
            "Using applicable scheme list: %d candidate schemes",
            len(applicable),
            extra={"call_id": context.call_id},
        )
        return applicable

    async def _evaluate(
        self,
        context: ConversationContext,
        model: str,
    ) -> EligibilityResult:
        """Get candidates, evaluate all batches, and merge the result."""
        candidate_schemes = self._get_candidate_schemes(context)
        batches = batch_schemes(candidate_schemes, self._batch_size)
        context.metadata["eligibility_candidate_count"] = len(candidate_schemes)
        context.metadata["eligibility_batch_count"] = len(batches)
        context.metadata["eligibility_missing_scheme_ids"] = []

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

        # Render the invariant prompt body (instructions + user profile) ONCE
        # per call. The schemes payload is the only per-batch variable, so we
        # leave a placeholder here and each batch substitutes its own payload.
        system_template = self._render_system_template(context)

        batch_results = await self._evaluate_batches(context, model, batches, system_template)
        matches_by_id: dict[str, SchemeMatch] = {}
        for result in batch_results:
            for match in result.matches:
                matches_by_id.setdefault(match.scheme_id, match)

        matches = [
            matches_by_id[scheme.scheme_id]
            for scheme in candidate_schemes
            if scheme.scheme_id in matches_by_id
        ]

        return EligibilityResult(
            matches=matches,
            processing_time_ms=0,
            model_used=model,
            schemes_evaluated=len(candidate_schemes),
        )

    def _render_system_template(self, context: ConversationContext) -> str:
        """Render the invariant system prompt once, leaving a schemes placeholder.

        The eligibility prompt embeds two variables: the user profile (constant
        across every batch of a single call) and the per-batch schemes payload.
        We render the profile here and keep ``{schemes}`` as a sentinel so each
        batch only has to splice in its own serialized schemes -- avoiding a
        full template re-render per batch.
        """
        profile_dict = context.user_profile.model_dump(mode="json", exclude_none=True)
        return prompts.render(
            "eligibility_system",
            user_profile=json_compact(profile_dict),
            schemes=_SCHEMES_PLACEHOLDER,
        )

    async def _evaluate_batches(
        self,
        context: ConversationContext,
        model: str,
        batches: list[list[SchemeRecord]],
        system_template: str,
    ) -> list[EligibilityResult]:
        """Evaluate scheme batches with bounded concurrency."""
        semaphore = asyncio.Semaphore(self._max_parallel_batches)

        async def _run(batch: list[SchemeRecord], batch_index: int) -> EligibilityResult:
            async with semaphore:
                return await self._evaluate_batch(
                    context, model, batch, batch_index, system_template
                )

        return await asyncio.gather(*[_run(batch, idx) for idx, batch in enumerate(batches)])

    async def _evaluate_batch(
        self,
        context: ConversationContext,
        model: str,
        batch: list[SchemeRecord],
        batch_index: int,
        system_template: str,
    ) -> EligibilityResult:
        """Evaluate one batch and retry once if the LLM omits scheme IDs."""
        result = await self._evaluate_batch_once(context, model, batch, system_template)
        result = self._normalize_batch_result(result, batch)
        missing = missing_candidate_ids(result.matches, batch)

        if missing:
            logger.warning(
                "Eligibility batch omitted schemes; retrying once",
                extra={
                    "batch_index": batch_index,
                    "missing_scheme_ids": missing,
                    "call_id": context.call_id,
                },
            )
            retry = await self._evaluate_batch_once(context, model, batch, system_template)
            retry = self._normalize_batch_result(retry, batch)
            result = self._merge_batch_retry(result, retry, batch)
            missing = missing_candidate_ids(result.matches, batch)

        if missing:
            context.metadata.setdefault("eligibility_missing_scheme_ids", []).extend(missing)
            result.matches.extend(
                uncertain_matches_for_missing(missing, batch, source="eligibility")
            )

        result.matches = normalize_matches_for_candidates(result.matches, batch)
        result.schemes_evaluated = len(batch)
        return result

    async def _evaluate_batch_once(
        self,
        context: ConversationContext,
        model: str,
        candidate_schemes: list[SchemeRecord],
        system_template: str,
    ) -> EligibilityResult:
        """Splice this batch's schemes into the pre-rendered prompt and call the LLM."""
        schemes_payload = self._serialize_schemes(candidate_schemes)
        system = system_template.replace(_SCHEMES_PLACEHOLDER, json_compact(schemes_payload))

        raw = await self._call_llm_json(
            system,
            "Evaluate eligibility now.",
            model=model,
            reasoning_effort=self._reasoning_effort,
            max_tokens=4096,
            wiki_grounding=True,
        )
        return self._parse_result(raw, model, len(candidate_schemes))

    @staticmethod
    def _normalize_batch_result(
        result: EligibilityResult,
        batch: list[SchemeRecord],
    ) -> EligibilityResult:
        result.matches = normalize_matches_for_candidates(result.matches, batch)
        return result

    @staticmethod
    def _merge_batch_retry(
        first: EligibilityResult,
        retry: EligibilityResult,
        batch: list[SchemeRecord],
    ) -> EligibilityResult:
        matches_by_id: dict[str, SchemeMatch] = {}
        for match in first.matches + retry.matches:
            matches_by_id.setdefault(match.scheme_id, match)
        first.matches = [
            matches_by_id[scheme.scheme_id]
            for scheme in batch
            if scheme.scheme_id in matches_by_id
        ]
        return first

    def _filter_schemes(self, user_state: str | None) -> list[SchemeRecord]:
        """Pre-filter schemes to those relevant to the user's state."""
        return filter_schemes_by_state(self._schemes, user_state)

    def _rank_schemes_by_retrieval(
        self,
        context: ConversationContext,
        applicable: list[SchemeRecord],
    ) -> list[SchemeRecord]:
        """Rank applicable schemes by RAG retrieval, pruned to the top-k hits.

        The retrieval-ranked top-k drives the LLM workload, so it stays flat as
        the corpus grows. Within one session the ranking is cached against a
        profile fingerprint, so repeated eligibility turns on an unchanged
        profile do not re-hit the vector store.
        """
        from vaidya.utils.states import state_name_to_code

        applicable_ids = {s.scheme_id for s in applicable}
        fingerprint = self._profile_fingerprint(context)

        cached = self._cached_retrieval_order(context, fingerprint, applicable)
        if cached is not None:
            logger.info(
                "Reusing cached RAG ranking (%d schemes); fingerprint unchanged",
                len(cached),
                extra={"call_id": context.call_id},
            )
            return cached

        retrieval_start = time.perf_counter()
        query = self._build_retrieval_query(context)
        state = context.user_profile.state
        state_code = state_name_to_code(state) if state else None

        hits = self._store.search(  # type: ignore[union-attr]
            query,
            n_results=self._retrieval_rank_top_k,
            state_code=state_code,
        )
        registry_map = {s.scheme_id: s for s in self._schemes}

        # Prune to the retrieval-ranked applicable hits (top-k), de-duplicated.
        # Schemes the user's state does not qualify for are never surfaced.
        seen: set[str] = set()
        ranked_hits: list[SchemeRecord] = []
        for stub in hits:
            resolved = registry_map.get(stub.scheme_id, stub)
            if resolved.scheme_id in applicable_ids and resolved.scheme_id not in seen:
                seen.add(resolved.scheme_id)
                ranked_hits.append(resolved)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        logger.info(
            "Scheme retrieval ranking completed",
            extra={
                "query_length": len(query),
                "state_code": state_code,
                "vector_hits": len(hits),
                "applicable_total": len(applicable),
                "ranked_hits": len(ranked_hits),
                "retrieval_ms": round(retrieval_ms, 1),
                "call_id": context.call_id,
            },
        )

        # Only cache a usable ordering; an empty result must not pin the cache,
        # so the caller can safely fall back to the full applicable set and a
        # later turn can retry retrieval.
        if ranked_hits:
            context.metadata[_RAG_CACHE_FINGERPRINT_KEY] = fingerprint
            context.metadata[_RAG_CACHE_ORDER_KEY] = [s.scheme_id for s in ranked_hits]
        return ranked_hits

    def _cached_retrieval_order(
        self,
        context: ConversationContext,
        fingerprint: str,
        applicable: list[SchemeRecord],
    ) -> list[SchemeRecord] | None:
        """Return the cached RAG ordering when the fingerprint is unchanged.

        The cached ordering is re-resolved against the *current* applicable set
        so any scheme that is no longer applicable (e.g. state changed) is
        dropped. Returns ``None`` when there is no valid cache hit, forcing a
        fresh retrieval.
        """
        if context.metadata.get(_RAG_CACHE_FINGERPRINT_KEY) != fingerprint:
            return None
        cached_order = context.metadata.get(_RAG_CACHE_ORDER_KEY)
        if not isinstance(cached_order, list) or not cached_order:
            return None

        applicable_by_id = {s.scheme_id: s for s in applicable}
        resolved = [
            applicable_by_id[scheme_id]
            for scheme_id in cached_order
            if scheme_id in applicable_by_id
        ]
        return resolved or None

    @staticmethod
    def _profile_fingerprint(context: ConversationContext) -> str:
        """Stable hash of the eligibility-relevant profile fields.

        Covers every field that drives the retrieval query or candidate set
        (``state``, ``occupation_type``, ``income_bracket``, ``health_need``),
        plus ``family_size`` and ``health_need_en`` as additional intake
        signals. Including the extra fields only makes the cache invalidate more
        eagerly when intake progresses -- never less -- so it cannot serve a
        stale ranking, while a turn that changes none of them reuses the cache.
        """
        profile = context.user_profile
        parts = [
            profile.state or "",
            profile.occupation_type.value,
            profile.income_bracket.value,
            str(profile.family_size) if profile.family_size is not None else "",
            profile.health_need or "",
            profile.health_need_en or "",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
        raw_matches = raw.get("matches", [])
        if not isinstance(raw_matches, list):
            raw_matches = []
        for item in raw_matches:
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
