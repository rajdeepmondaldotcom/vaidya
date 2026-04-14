"""Generate a Markdown evaluation report from VaidyaEvaluator results.

The report is written to ``eval_report.md`` in the working directory unless
a different path is supplied.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from eval.evaluator import EvalSummary, ScenarioResult


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def generate_report(
    results: list[ScenarioResult],
    summary: EvalSummary,
    output_path: str | Path = "eval_report.md",
) -> str:
    """Build a Markdown report and write it to *output_path*.

    Returns the report text.
    """
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("# Vaidya Evaluation Report")
    lines.append(f"\nGenerated: {now}")
    lines.append("")

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total scenarios | {summary.total} |")
    lines.append(f"| Passed | {summary.passed} ({_pct(summary.passed, summary.total)}) |")
    lines.append(f"| Failed | {summary.failed} ({_pct(summary.failed, summary.total)}) |")
    lines.append(f"| Errors (API / timeout) | {summary.errors} |")
    lines.append(f"| Average latency | {summary.avg_latency_ms:.0f} ms |")
    lines.append("")

    # ------------------------------------------------------------------
    # Per-scenario results table
    # ------------------------------------------------------------------
    lines.append("## Per-Scenario Results")
    lines.append("")
    lines.append(
        "| ID | Name | Result | Eligible (actual) | False Neg | False Pos | Latency | Phase |"
    )
    lines.append("|----|----|--------|----|----|----|----|------|")

    for r in results:
        status = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
        eligible_str = ", ".join(r.eligible_schemes) if r.eligible_schemes else "none"
        fn_str = ", ".join(r.false_negatives) if r.false_negatives else "-"
        fp_str = ", ".join(r.false_positives) if r.false_positives else "-"
        lines.append(
            f"| {r.scenario_id} | {r.scenario_name} | **{status}** | "
            f"{eligible_str} | {fn_str} | {fp_str} | "
            f"{r.latency_ms:.0f} ms | {r.final_phase} |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Per-scheme accuracy
    # ------------------------------------------------------------------
    lines.append("## Per-Scheme Accuracy")
    lines.append("")

    all_scheme_ids = sorted(
        set(summary.per_scheme_tp) | set(summary.per_scheme_fn) | set(summary.per_scheme_fp)
    )

    if all_scheme_ids:
        lines.append("| Scheme ID | True Pos | False Neg | False Pos | Recall |")
        lines.append("|-----------|----------|-----------|-----------|--------|")

        for sid in all_scheme_ids:
            tp = summary.per_scheme_tp.get(sid, 0)
            fn = summary.per_scheme_fn.get(sid, 0)
            fp = summary.per_scheme_fp.get(sid, 0)
            recall = _pct(tp, tp + fn)
            lines.append(f"| {sid} | {tp} | {fn} | {fp} | {recall} |")
        lines.append("")
    else:
        lines.append("No per-scheme data collected (all scenarios may have errored).\n")

    # ------------------------------------------------------------------
    # Reviewer agreement rate
    # ------------------------------------------------------------------
    lines.append("## Reviewer Agreement")
    lines.append("")

    # Infer reviewer agreement from false positives on exclusion tests.
    # If an expected-ineligible scheme appears as eligible, the reviewer missed it.
    exclusion_scenarios = [r for r in results if r.expected_ineligible and not r.error]
    if exclusion_scenarios:
        caught = sum(1 for r in exclusion_scenarios if not r.false_positives)
        total_excl = len(exclusion_scenarios)
        lines.append(
            f"Scenarios testing exclusion rules: **{total_excl}**  \n"
            f"Correctly excluded: **{caught}** ({_pct(caught, total_excl)})  "
        )
        lines.append("")

        missed = [r for r in exclusion_scenarios if r.false_positives]
        if missed:
            lines.append("### Missed Exclusions")
            lines.append("")
            for r in missed:
                lines.append(
                    f"- **{r.scenario_id}** ({r.scenario_name}): "
                    f"false positives = {', '.join(r.false_positives)}"
                )
            lines.append("")
    else:
        lines.append("No exclusion-rule scenarios available for reviewer agreement analysis.\n")

    # ------------------------------------------------------------------
    # Failure details
    # ------------------------------------------------------------------
    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("## Failure Details")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.scenario_id}: {r.scenario_name}")
            lines.append("")
            if r.error:
                lines.append(f"**Error:** `{r.error}`")
            if r.false_negatives:
                lines.append(f"**Missing eligible schemes:** {', '.join(r.false_negatives)}")
            if r.false_positives:
                lines.append(
                    f"**Incorrectly eligible (should be excluded):** "
                    f"{', '.join(r.false_positives)}"
                )
            lines.append("")
            # Show last few conversation turns for debugging
            if r.conversation:
                lines.append("<details><summary>Conversation excerpt (last 4 turns)</summary>\n")
                for turn in r.conversation[-4:]:
                    role = turn.get("role", "?")
                    text = turn.get("text", "")
                    lines.append(f"**{role}:** {text}\n")
                lines.append("</details>\n")

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    lines.append("## Recommendations")
    lines.append("")

    recs: list[str] = []

    if summary.errors > 0:
        recs.append(
            f"- **Fix API connectivity:** {summary.errors} scenario(s) failed "
            f"with transport or timeout errors."
        )

    # Check recall per scheme
    for sid in all_scheme_ids:
        tp = summary.per_scheme_tp.get(sid, 0)
        fn = summary.per_scheme_fn.get(sid, 0)
        if tp + fn > 0 and tp / (tp + fn) < 0.8:
            recs.append(
                f"- **Improve {sid} recall:** currently {_pct(tp, tp + fn)}. "
                f"Check intake extraction for fields this scheme depends on."
            )

    # Check false positive rate
    for sid in all_scheme_ids:
        fp = summary.per_scheme_fp.get(sid, 0)
        if fp > 0:
            recs.append(
                f"- **{sid} false positive:** appeared eligible in {fp} scenario(s) "
                f"where it should have been excluded. Review exclusion rule handling."
            )

    if summary.avg_latency_ms > 30_000:
        recs.append(
            f"- **Reduce latency:** average {summary.avg_latency_ms:.0f} ms per scenario. "
            f"Target < 30 s for voice-acceptable response times."
        )

    # Cross-language parity check
    hi_result = next((r for r in results if r.scenario_id == "SC-V013-HI"), None)
    ta_result = next((r for r in results if r.scenario_id == "SC-V013-TA"), None)
    if (
        hi_result
        and ta_result
        and set(hi_result.eligible_schemes) != set(ta_result.eligible_schemes)
    ):
        recs.append(
            "- **Cross-language parity gap:** SC-V013-HI and SC-V013-TA produced "
            "different eligibility results for the same profile. "
            "Investigate multilingual intake extraction."
        )

    if not recs:
        recs.append("- All checks passed. Continue expanding scenario coverage.")

    lines.extend(recs)
    lines.append("")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")
    return report_text
