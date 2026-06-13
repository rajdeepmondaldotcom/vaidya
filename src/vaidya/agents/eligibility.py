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
    EligibilityVerdict,
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

# Deterministic floor for PM-JAY, the flagship national scheme. The eligibility
# LLM intermittently DROPS PM-JAY for clearly-qualifying low-income informal
# workers (it treats the SECC 2011 D-code as a hard requirement the caller can
# never recite), so the same Rajasthan daily-wage family hears about PM-JAY on
# one call and not the next. These are the OBJECTIVE inclusion criteria from the
# scheme record; when the structured profile clearly meets them we PROPOSE
# PM-JAY eligible and let the Reviewer veto on any passing exclusion.
_PMJAY_CENTRAL_ID_PREFIX = "PMJAY-2024"  # the central flagship, not state variants
# Occupations unambiguously inside PM-JAY's occupation_included list:
_PMJAY_FLOOR_OCCUPATIONS = frozenset({"daily_wage", "farmer"})
_PMJAY_FLOOR_INCOMES = frozenset({"below_1l", "1l_to_2.5l"})  # below the Rs 2.5L SECC ceiling
_PMJAY_BLOCKING_COVERAGE = frozenset({"employer", "private"})  # hard PM-JAY exclusion

# Placeholder substituted into the rendered system prompt once per call so the
# invariant prompt body (instructions + user profile) is rendered a single time
# and each batch only swaps in its own serialized scheme payload.
_SCHEMES_PLACEHOLDER = "\x00__VAIDYA_SCHEMES_PAYLOAD__\x00"

# Keys used to cache the RAG retrieval ordering across turns of one session.
_RAG_CACHE_FINGERPRINT_KEY = "eligibility_rag_fingerprint"
_RAG_CACHE_ORDER_KEY = "eligibility_rag_scheme_id_order"


# Hard cap on the number of in-flight/finished speculative entries retained on
# the (app-singleton) agent. Sessions that start speculation but never reach a
# consume/cancel path (Redis TTL expiry, mid-intake hang-up) would otherwise
# leak entries forever; mirrors ConversationManager's _turn_locks pruning.
_MAX_SPECULATIVE_ENTRIES = 1000


class _SpeculativeEntry:
    """A speculative eligibility computation kicked off during intake.

    Holds the *full-profile* eligibility-input fingerprint the speculation was
    started under plus the in-flight :class:`asyncio.Task`. The fingerprint is
    the safety gate: a speculative result is reused ONLY when the *current*
    fingerprint still matches ``fingerprint``. Any mismatch (the profile changed
    after speculation began) means the result is stale and MUST be discarded in
    favour of a fresh synchronous computation.

    The fingerprint MUST cover every profile field the eligibility evaluation
    depends on -- i.e. the entire serialised profile fed to the prompt -- NOT
    just the narrow RAG-retrieval fingerprint, otherwise a change to a field the
    LLM uses (coverage, age, BPL/ration/SECC, district) but the RAG fingerprint
    omits would silently reuse a stale verdict. See
    :meth:`EligibilityAgent._eligibility_input_fingerprint`.

    Task handles and results live on the agent instance (keyed by ``call_id``),
    never on ``ConversationContext.metadata`` -- the context is JSON-serialised
    into Redis between turns and an ``asyncio.Task`` is not serialisable. Because
    speculation is a pure latency optimisation, a missing entry (e.g. a turn
    served by a different worker process) simply falls back to computing fresh.
    """

    __slots__ = ("fingerprint", "task")

    def __init__(
        self,
        fingerprint: str,
        task: asyncio.Task[EligibilityResult],
    ) -> None:
        self.fingerprint = fingerprint
        self.task = task


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
        # Speculative eligibility passes kicked off during intake, keyed by
        # call_id. Lives on the (long-lived, app-singleton) agent instance --
        # NOT on the serialised ConversationContext -- because it holds live
        # asyncio.Task objects. See _SpeculativeEntry.
        self._speculative: dict[str, _SpeculativeEntry] = {}

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Run eligibility evaluation against the scheme corpus.

        If a speculative pass was kicked off during intake (see
        :meth:`start_speculative`) and its fingerprint still matches the
        current profile, its result is reused instead of recomputing -- this is
        the latency win, with byte-identical output to a fresh pass. On any
        fingerprint mismatch, speculative failure, or absent entry we fall
        through to a fresh synchronous computation. Correctness is therefore
        identical to never having speculated.
        """
        start_ms = time.perf_counter()

        result, model_used = await self._resolve_result(context)
        if result is None:
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
                "eligibility_speculative_hit": context.metadata.get(
                    "eligibility_speculative_hit",
                    False,
                ),
            },
        )

    async def _resolve_result(
        self,
        context: ConversationContext,
    ) -> tuple[EligibilityResult | None, str]:
        """Return (result, model_used), reusing a valid speculation if present.

        Returns ``(None, model)`` only when both the primary and fallback
        models fail a fresh computation -- the caller then emits the spoken
        fallback. The speculative path can never produce ``None``: a failed or
        mismatched speculation is discarded and we recompute synchronously.
        """
        speculative = await self._consume_speculative(context)
        if speculative is not None:
            context.metadata["eligibility_speculative_hit"] = True
            logger.info(
                "Reusing speculative eligibility result (%d matches); fingerprint unchanged",
                len(speculative.matches),
                extra={"call_id": context.call_id},
            )
            return speculative, speculative.model_used

        context.metadata["eligibility_speculative_hit"] = False
        return await self._compute_with_fallback(context)

    async def _compute_with_fallback(
        self,
        context: ConversationContext,
    ) -> tuple[EligibilityResult | None, str]:
        """Evaluate with the primary model, falling back to the secondary once."""
        model_used = self._model
        try:
            return await self._evaluate(context, model_used), model_used
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
                return await self._evaluate(context, model_used), model_used
            except Exception as fallback_exc:
                logger.error(
                    "Eligibility evaluation failed on both models",
                    extra={"error": str(fallback_exc), "call_id": context.call_id},
                    exc_info=True,
                )
                return None, model_used

    # ------------------------------------------------------------------
    # Speculative execution (started during intake, consumed at PROCESSING)
    # ------------------------------------------------------------------

    def start_speculative(self, context: ConversationContext) -> bool:
        """Kick off a non-blocking eligibility pass for the current profile.

        Intended to be called once enough of the profile is known to evaluate
        (by the last intake question). Spawns a background :class:`asyncio.Task`
        that computes the full eligibility result and stashes it on the agent
        instance keyed by ``call_id``. The conversation turn is NOT blocked.

        Returns ``True`` when a (new or already-running) speculation covers the
        current fingerprint, ``False`` when nothing was started (no event loop,
        profile not yet evaluable, or an unexpected error). A ``False`` return
        is harmless -- PROCESSING just computes synchronously.

        Idempotent per fingerprint: a second call with an unchanged profile
        reuses the in-flight task; a call after the profile changed cancels the
        now-stale task and starts a fresh one.
        """
        try:
            if not self._profile_is_evaluable(context):
                return False

            # Gate on the FULL eligibility-input fingerprint, not the narrow RAG
            # one: the reused result must be invalidated by a change to ANY
            # profile field the LLM evaluation reads.
            fingerprint = self._eligibility_input_fingerprint(context)
            existing = self._speculative.get(context.call_id)
            if existing is not None and existing.fingerprint == fingerprint:
                # Same profile already being (or done being) speculated -- keep it.
                return True

            # Profile changed (or first time): drop any stale in-flight task.
            if existing is not None:
                self._cancel_entry(existing)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop (e.g. called from sync context) -- skip.
                return False

            self._prune_speculative()

            # Operate on a deep copy so the background task NEVER mutates the
            # live context (which is being serialised to Redis at the end of
            # this turn and reloaded fresh on the next). This snapshots the
            # exact profile the fingerprint was taken over and gives the task
            # its own metadata scratch space -- no races, no cross-turn bleed.
            snapshot = context.model_copy(deep=True)
            task: asyncio.Task[EligibilityResult] = loop.create_task(
                self._speculative_evaluate(snapshot),
                name=f"speculative-eligibility-{context.call_id}",
            )
            # Swallow speculative-path exceptions so a failed background task can
            # never surface as an unhandled-exception warning; correctness falls
            # back to the synchronous path. The done-callback also clears a
            # finished task that nobody consumed.
            task.add_done_callback(self._on_speculative_done)
            self._speculative[context.call_id] = _SpeculativeEntry(fingerprint, task)
            logger.info(
                "Started speculative eligibility pass during intake",
                extra={"call_id": context.call_id},
            )
            return True
        except Exception:
            logger.warning(
                "Failed to start speculative eligibility; will compute synchronously",
                extra={"call_id": context.call_id},
                exc_info=True,
            )
            return False

    async def _consume_speculative(
        self,
        context: ConversationContext,
    ) -> EligibilityResult | None:
        """Return a speculative result IFF its fingerprint matches the profile.

        Pops the entry regardless of outcome (it is single-use). A fingerprint
        mismatch, a cancelled task, or a task that raised all yield ``None`` so
        the caller recomputes synchronously. The reviewer + convergence path is
        unaffected -- this only swaps in an identical eligibility result.
        """
        entry = self._speculative.pop(context.call_id, None)
        if entry is None:
            return None

        # SAFETY GATE: every eligibility-relevant profile field must be
        # byte-for-byte the same as when the speculation started, else the
        # speculative result is stale. Uses the FULL profile fingerprint.
        current_fingerprint = self._eligibility_input_fingerprint(context)
        if entry.fingerprint != current_fingerprint:
            logger.info(
                "Discarding speculative eligibility: profile fingerprint changed",
                extra={"call_id": context.call_id},
            )
            self._cancel_entry(entry)
            return None

        try:
            result = await entry.task
        except asyncio.CancelledError:
            return None
        except Exception:
            logger.warning(
                "Speculative eligibility task raised; computing synchronously",
                extra={"call_id": context.call_id},
                exc_info=True,
            )
            return None

        if not isinstance(result, EligibilityResult):
            return None
        return result

    def cancel_speculative(self, call_id: str) -> None:
        """Cancel and drop any in-flight speculation for a session.

        Called on session end / abandonment so background tasks never leak.
        Safe to call when nothing is in flight.
        """
        entry = self._speculative.pop(call_id, None)
        if entry is not None:
            self._cancel_entry(entry)

    async def _speculative_evaluate(
        self,
        context: ConversationContext,
    ) -> EligibilityResult:
        """Background body of a speculative pass: full evaluation with fallback.

        Mirrors the synchronous compute path (primary model, then fallback) and
        stamps timing/model onto the result so a reuse hit is indistinguishable
        from a fresh pass. Exceptions propagate to the awaiting consumer, which
        then recomputes synchronously.
        """
        start_ms = time.perf_counter()
        result, model_used = await self._compute_with_fallback(context)
        if result is None:
            # Both models failed during speculation. Raise so the consumer's
            # await sees it and falls back to a fresh synchronous attempt.
            raise RuntimeError("speculative eligibility evaluation failed on both models")
        result.processing_time_ms = round((time.perf_counter() - start_ms) * 1000, 1)
        result.model_used = model_used
        return result

    def _profile_is_evaluable(self, context: ConversationContext) -> bool:
        """True when the profile has the fields eligibility needs to run.

        Mirrors :pyattr:`UserProfile.required_fields_complete` (state, family
        size, income, occupation, coverage all known) -- i.e. everything the
        five intake questions collect. Below this bar a speculative pass would
        run against an incomplete profile and its fingerprint would not match
        the post-intake profile anyway, so we simply do not start one.
        """
        return context.user_profile.required_fields_complete

    @staticmethod
    def _eligibility_input_fingerprint(context: ConversationContext) -> str:
        """Stable hash of the ENTIRE profile the eligibility evaluation reads.

        Unlike :meth:`_profile_fingerprint` (scoped to the RAG query / candidate
        set), this must change whenever ANY field that can affect a verdict
        changes, because the full profile is serialised into the eligibility
        prompt (coverage, age, BPL/ration/SECC, district all feed the LLM's
        criteria). It is the safety gate for reusing a cached *result*: identical
        fingerprint => identical eligibility input => identical verdict.

        We hash the same canonical JSON that :meth:`_render_system_template`
        feeds the prompt, so the gate tracks the prompt input exactly.
        """
        profile_dict = context.user_profile.model_dump(mode="json", exclude_none=True)
        raw = json_compact(profile_dict)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prune_speculative(self) -> None:
        """Bound the speculative map so abandoned sessions can't leak forever.

        Entries are normally removed on consume (PROCESSING) or cancel (session
        end). A session that starts speculation but never reaches either path
        (Redis TTL expiry, silent hang-up) would otherwise retain its entry for
        the lifetime of this app-singleton agent. Called just before inserting a
        new entry: evicts the oldest entries (insertion-ordered dict) so that
        after the imminent insert the map holds at most ``_MAX_SPECULATIVE_ENTRIES``.
        """
        # Leave room for the one entry about to be inserted by the caller.
        overflow = len(self._speculative) - (_MAX_SPECULATIVE_ENTRIES - 1)
        if overflow <= 0:
            return
        for call_id in list(self._speculative.keys())[:overflow]:
            entry = self._speculative.pop(call_id, None)
            if entry is not None:
                self._cancel_entry(entry)

    @staticmethod
    def _cancel_entry(entry: _SpeculativeEntry) -> None:
        if not entry.task.done():
            entry.task.cancel()

    @staticmethod
    def _on_speculative_done(task: asyncio.Task[EligibilityResult]) -> None:
        """Consume any exception/cancellation so it is never 'never retrieved'."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug("Speculative eligibility task ended with error: %s", exc)

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
        matches = self._apply_flagship_floor(matches, candidate_schemes, context.user_profile)

        return EligibilityResult(
            matches=matches,
            processing_time_ms=0,
            model_used=model,
            schemes_evaluated=len(candidate_schemes),
        )

    def _apply_flagship_floor(
        self,
        matches: list[SchemeMatch],
        candidate_schemes: list[SchemeRecord],
        profile: Any,
    ) -> list[SchemeMatch]:
        """Ensure PM-JAY is PROPOSED eligible when the profile objectively meets
        its criteria, so the flagship is never flakily dropped by the LLM.

        This only proposes -- it never overrides a hard exclusion (the criteria
        themselves exclude govt employees, employer-insured, and WB/DL), and the
        Reviewer Agent still reads the full transcript and can veto on an
        exclusion mentioned in passing. So the dual-check safety is preserved;
        we just stop the LLM's intermittent recall misses on a clear-cut case.
        """
        pmjay = next(
            (s for s in candidate_schemes if s.scheme_id.startswith(_PMJAY_CENTRAL_ID_PREFIX)),
            None,
        )
        if pmjay is None or not self._pmjay_floor_eligible(profile, pmjay):
            return matches

        floored = SchemeMatch(
            scheme_id=pmjay.scheme_id,
            scheme_name=pmjay.canonical_name,
            verdict=EligibilityVerdict.ELIGIBLE,
            confidence=0.8,
            reasoning_trace="",
            matched_criteria=["state", "income_bracket", "occupation_type"],
            failed_criteria=[],
            coverage_summary="Rs 5 lakh family cover per year",
        )
        for index, match in enumerate(matches):
            if match.scheme_id == pmjay.scheme_id:
                if match.verdict != EligibilityVerdict.ELIGIBLE:
                    logger.info(
                        "Flagship floor: upgrading PM-JAY to eligible (objective criteria met)",
                        extra={"prior_verdict": match.verdict.value},
                    )
                    matches[index] = floored
                return matches
        matches.append(floored)
        return matches

    @staticmethod
    def _pmjay_floor_eligible(profile: Any, record: SchemeRecord) -> bool:
        """True when the structured profile unambiguously satisfies PM-JAY's
        objective inclusion criteria (state, income, occupation, coverage)."""
        from vaidya.utils.states import state_name_to_code

        state_code = state_name_to_code(profile.state) if profile.state else None
        if not state_code or state_code in (record.geographic_restrictions or []):
            return False  # state unknown, or a PM-JAY opt-out state (WB/DL)
        if profile.income_bracket.value not in _PMJAY_FLOOR_INCOMES:
            return False
        if profile.occupation_type.value not in _PMJAY_FLOOR_OCCUPATIONS:
            return False
        return profile.existing_coverage.value not in _PMJAY_BLOCKING_COVERAGE

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
