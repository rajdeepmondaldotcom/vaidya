"""CLI entry point for the Vaidya evaluation framework.

Usage
-----
    # Run all scenarios against a local server
    python -m eval --base-url http://localhost:8000

    # Run the quick smoke-test subset
    python -m eval --scenarios quick

    # Run specific scenarios by ID
    python -m eval --scenarios SC-V001,SC-V003,SC-V011

    # Run scenarios matching a tag
    python -m eval --tag adversarial

    # Increase concurrency (default 1 -- sequential)
    python -m eval --concurrency 3

    # Custom report output path
    python -m eval --output results/eval_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from eval.evaluator import VaidyaEvaluator
from eval.report import generate_report
from eval.scenarios import (
    get_all_scenarios,
    get_quick_scenarios,
    get_scenario_by_id,
    get_scenarios_by_tag,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vaidya-eval",
        description="Run Vaidya evaluation scenarios and generate a report.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running Vaidya API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Which scenarios to run. Options: "
            "'all' (default), 'quick' (smoke test), "
            "or comma-separated IDs like 'SC-V001,SC-V003'"
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Run only scenarios matching this tag (e.g. 'adversarial', 'hindi', 'exclusion')",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent scenario runs (default: 1 -- sequential)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per scenario in seconds (default: 120)",
    )
    parser.add_argument(
        "--output",
        default="eval_report.md",
        help="Path for the Markdown report output (default: eval_report.md)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def _resolve_scenarios(args: argparse.Namespace) -> list[dict]:
    """Resolve the --scenarios / --tag arguments into a concrete scenario list."""
    # Tag filter takes priority if provided alongside scenarios
    if args.tag:
        scenarios = get_scenarios_by_tag(args.tag)
        if not scenarios:
            print(f"No scenarios found with tag '{args.tag}'", file=sys.stderr)
            sys.exit(1)
        return scenarios

    choice = args.scenarios.strip().lower()

    if choice == "all":
        return get_all_scenarios()

    if choice == "quick":
        return get_quick_scenarios()

    # Comma-separated IDs
    ids = [s.strip() for s in args.scenarios.split(",")]
    scenarios = []
    for sid in ids:
        sc = get_scenario_by_id(sid)
        if sc is None:
            print(f"Unknown scenario ID: {sid}", file=sys.stderr)
            sys.exit(1)
        scenarios.append(sc)
    return scenarios


async def _main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    scenarios = _resolve_scenarios(args)
    print(f"\nVaidya Eval: running {len(scenarios)} scenario(s) against {args.base_url}\n")

    evaluator = VaidyaEvaluator(
        api_base_url=args.base_url,
        timeout=args.timeout,
    )

    results = await evaluator.run_all(scenarios, concurrency=args.concurrency)
    summary = VaidyaEvaluator.summarize(results)

    # Print quick summary to stdout
    print("\n" + "=" * 60)
    print(
        f"  PASS: {summary.passed}/{summary.total}    "
        f"FAIL: {summary.failed}/{summary.total}    "
        f"ERRORS: {summary.errors}"
    )
    print(f"  Avg latency: {summary.avg_latency_ms:.0f} ms")
    print("=" * 60)

    # Generate report
    generate_report(results, summary, output_path=args.output)
    print(f"\nReport written to {args.output}")

    # Exit with non-zero if any failures
    if summary.failed > 0:
        sys.exit(1)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
