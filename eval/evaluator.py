"""VaidyaEvaluator: run test scenarios against the Vaidya API and score results.

Sends multi-turn conversations to the /simulate/text endpoint and compares
the returned eligible_schemes against each scenario's expected outcomes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Outcome of running a single evaluation scenario."""

    scenario_id: str
    scenario_name: str
    passed: bool
    eligible_schemes: list[str]
    expected_eligible: list[str]
    expected_ineligible: list[str]
    false_negatives: list[str]  # expected eligible but missing
    false_positives: list[str]  # expected ineligible but returned
    latency_ms: float
    final_phase: str
    conversation: list[dict[str, str]]
    error: str | None = None


@dataclass
class EvalSummary:
    """Aggregate statistics across all evaluated scenarios."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    per_scheme_tp: dict[str, int] = field(default_factory=dict)
    per_scheme_fn: dict[str, int] = field(default_factory=dict)
    per_scheme_fp: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class VaidyaEvaluator:
    """Drives multi-turn test scenarios through the Vaidya simulation API."""

    DEFAULT_TIMEOUT = 120.0  # seconds -- multi-turn calls can be slow

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = api_base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Single scenario
    # ------------------------------------------------------------------

    async def run_scenario(self, scenario: dict[str, Any]) -> ScenarioResult:
        """Run one scenario and return a scored result."""
        scenario_id: str = scenario["id"]
        scenario_name: str = scenario["name"]
        expected_eligible: list[str] = scenario["expected_eligible_schemes"]
        expected_ineligible: list[str] = scenario["expected_ineligible_schemes"]

        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/simulate/text",
                    json={
                        "turns": scenario["turns"],
                        "language": scenario.get("language", "hi-IN"),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Scenario %s failed with error: %s",
                scenario_id,
                exc,
            )
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                passed=False,
                eligible_schemes=[],
                expected_eligible=expected_eligible,
                expected_ineligible=expected_ineligible,
                false_negatives=expected_eligible,
                false_positives=[],
                latency_ms=elapsed,
                final_phase="error",
                conversation=[],
                error=str(exc),
            )

        elapsed = (time.perf_counter() - start) * 1000

        eligible_schemes: list[str] = data.get("eligible_schemes", [])
        conversation: list[dict[str, str]] = data.get("conversation", [])
        final_phase: str = data.get("final_phase", "unknown")

        # Score: check expected eligible appear in results
        eligible_set = set(eligible_schemes)
        false_negatives = [s for s in expected_eligible if s not in eligible_set]
        # Score: check expected ineligible do NOT appear
        false_positives = [s for s in expected_ineligible if s in eligible_set]

        passed = len(false_negatives) == 0 and len(false_positives) == 0

        result = ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            passed=passed,
            eligible_schemes=eligible_schemes,
            expected_eligible=expected_eligible,
            expected_ineligible=expected_ineligible,
            false_negatives=false_negatives,
            false_positives=false_positives,
            latency_ms=elapsed,
            final_phase=final_phase,
            conversation=conversation,
        )

        status = "PASS" if passed else "FAIL"
        logger.info(
            "[%s] %s  (%s)  %.0f ms",
            status,
            scenario_id,
            scenario_name,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    async def run_all(
        self,
        scenarios: list[dict[str, Any]],
        concurrency: int = 1,
    ) -> list[ScenarioResult]:
        """Run all scenarios, respecting *concurrency* limit.

        Sequential by default (concurrency=1) to avoid overwhelming the API.
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: list[ScenarioResult] = []

        async def _run_one(scenario: dict[str, Any]) -> ScenarioResult:
            async with semaphore:
                return await self.run_scenario(scenario)

        tasks = [_run_one(s) for s in scenarios]
        results = await asyncio.gather(*tasks)
        return list(results)

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(results: list[ScenarioResult]) -> EvalSummary:
        """Compute aggregate statistics from a batch of results."""
        summary = EvalSummary()
        summary.total = len(results)
        total_latency = 0.0

        for r in results:
            total_latency += r.latency_ms

            if r.error:
                summary.errors += 1
                summary.failed += 1
                continue

            if r.passed:
                summary.passed += 1
            else:
                summary.failed += 1

            # Per-scheme scoring
            eligible_set = set(r.eligible_schemes)
            for scheme_id in r.expected_eligible:
                if scheme_id in eligible_set:
                    summary.per_scheme_tp[scheme_id] = summary.per_scheme_tp.get(scheme_id, 0) + 1
                else:
                    summary.per_scheme_fn[scheme_id] = summary.per_scheme_fn.get(scheme_id, 0) + 1
            for scheme_id in r.expected_ineligible:
                if scheme_id in eligible_set:
                    summary.per_scheme_fp[scheme_id] = summary.per_scheme_fp.get(scheme_id, 0) + 1

        summary.avg_latency_ms = total_latency / max(summary.total, 1)
        return summary
