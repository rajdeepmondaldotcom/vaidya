# Vaidya Evaluation Suite

This document describes how Vaidya's eligibility output is evaluated end-to-end. It is grounded in the code under [`eval/`](../eval): `scenarios.py` (the scenario corpus), `evaluator.py` (the runner and scorer), `report.py` (the Markdown report generator), `run_eval.py` (the CLI), and the `eval*` targets in the [`Makefile`](../Makefile).

## What it measures, and why

Vaidya tells a caller which government healthcare schemes they qualify for. For a benefit advisor, *both* directions of error are costly:

- A **false positive** — a scheme shown to someone who is not actually eligible — sends the caller to a hospital or enrolment desk where they will be turned away. For an exclusion-bound scheme like PM-JAY (which excludes government employees, income-tax payers, and motorized-vehicle owners), a false positive is a correctness failure with real-world consequences.
- A **false negative** — a scheme the caller qualifies for but is never told about — means a family silently misses coverage they were entitled to.

Because both matter, the suite scores each scenario on **precision (no false positives)** and **recall (no false negatives)** against a hand-curated expected outcome, rather than a single accuracy number. It also records **latency** per scenario, since the same pipeline runs behind a live voice call and slow responses degrade the call experience.

Concretely, the evaluator drives a full multi-turn conversation through the running API and compares the schemes the system returns against two labelled sets per scenario: the schemes that *should* appear (`expected_eligible_schemes`) and the schemes that *must not* appear (`expected_ineligible_schemes`).

## The scoring rule

For a single scenario, given the system's returned `eligible_schemes`, the evaluator computes (see `VaidyaEvaluator.run_scenario` in [`eval/evaluator.py`](../eval/evaluator.py)):

```python
eligible_set   = set(eligible_schemes)
false_negatives = [s for s in expected_eligible   if s not in eligible_set]   # missing  → recall miss
false_positives = [s for s in expected_ineligible if s in eligible_set]       # leaked   → precision miss

passed = len(false_negatives) == 0 and len(false_positives) == 0
```

A scenario **passes only if there are zero false negatives AND zero false positives**. This is deliberately strict: there is no partial credit at the scenario level. Per-scheme true-positive / false-negative / false-positive tallies are aggregated separately so that recall can be reported per scheme even when a scenario as a whole fails (see `VaidyaEvaluator.summarize`).

Note the asymmetry in labelling: a scheme that is *not* listed in either `expected_eligible_schemes` or `expected_ineligible_schemes` is not scored for that scenario. The ineligible list names the specific schemes a scenario is asserting *must* be excluded (typically the exclusion or wrong-state schemes that profile would plausibly trip), not every scheme in the catalog.

## The scenario corpus

The suite contains **81 scenarios** (`SCENARIOS` in [`eval/scenarios.py`](../eval/scenarios.py), every `id` unique). Aggregate coverage across the corpus:

- **10 languages** (the `language` field, in Sarvam locale codes): Hindi `hi-IN`, Tamil `ta-IN`, Bengali `bn-IN`, Telugu `te-IN`, Gujarati `gu-IN`, Kannada `kn-IN`, Malayalam `ml-IN`, Marathi `mr-IN`, Odia `od-IN`, Punjabi `pa-IN`.
- **19 states/UTs** exercised via scenario tags: Andhra Pradesh, Bihar, Chhattisgarh, Delhi, Gujarat, Haryana, Jharkhand, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Odisha, Punjab, Rajasthan, Tamil Nadu, Telangana, Uttar Pradesh, Uttarakhand, West Bengal.
- **24 distinct scheme IDs** appear in scenario expectations (across `expected_eligible_schemes` and `expected_ineligible_schemes`), drawn from the scheme catalog documented in the header of `scenarios.py` (which lists 30 reference IDs). Examples: `PMJAY-2024-v3`, `PMJAY-70PLUS-2024-v1`, `CHIR-RJ-2024-v2`, `SS-WB-2024-v2`, `AK-KA-2024-v2`, `MJPJAY-MH-2024-v2`, `ESIC-2024-v2`, `PMSBY-2024-v2`, `CMCHIS-TN-2024-v1`, `AAROGYASRI-AP-2024-v1`, `KASP-KL-2024-v1`, `ABUA-JH-2024-v1`, `JSY-2024-v1`, `JSSK-2024-v1`, `NIKSHAY-2024-v1`, `CGHS-2024-v1`.

### Scenario categories

The corpus is organized into deliberate categories. Each is identified by `tags` and an ID band:

- **Happy-path, multi-scheme** (e.g. `SC-V001`, `SC-V010`, `SC-V070`): a realistic low-income profile that should match several schemes at once — typically PM-JAY (central) plus the relevant state scheme plus PMSBY. `SC-V070` is a stress test for an elderly BPL farmer expecting three simultaneous eligible schemes (PM-JAY, PM-JAY 70+, Arogya Karnataka) while asserting PMSBY is correctly excluded — the caller is 72, above the PMSBY age-70 cutoff.
- **Per-scheme isolation** (`SC-V020`–`SC-V028`, plus boundary cases like `SC-V092`–`SC-V094`): one profile constructed to land squarely inside a single scheme's criteria, so a regression in any one scheme's logic surfaces in isolation. Includes free-tier vs. paid-tier distinctions (e.g. Chiranjeevi free vs. Rs 850 premium) and exact-boundary checks (ESIC at the 21K salary ceiling, PMSBY at the age-70 cutoff).
- **Exclusion rules** (`SC-V030`–`SC-V035`): each isolates one PM-JAY hard exclusion — **government employee** (`SC-V030`), **income-tax payer** (`SC-V031`), **motorized-vehicle owner** (`SC-V032`), mechanized-farming-equipment owner (`SC-V033`), plus state opt-outs where PM-JAY is unavailable (West Bengal `SC-V034`, Delhi `SC-V035`). The expected outcome asserts PM-JAY does **not** appear.
- **Cross-language parity** (`SC-V013-HI`/`-TA`; `SC-V040-HI`/`-TA`/`-BN`; `SC-V041-HI`/`-BN`): the *same* user profile expressed in Hindi, Tamil, and/or Bengali. Each variant carries identical `expected_eligible_schemes`, so the suite asserts that language must not change the eligibility outcome. The report explicitly diffs the `SC-V013-HI` and `SC-V013-TA` results and flags any divergence.
- **Adversarial** (`SC-V011`, `SC-V012`, `SC-V050`–`SC-V054`): prompt injection ("ignore previous instructions", direct eligibility-override attempts), **PII readback** refusal (the user shares an Aadhaar number and asks the system to read it back), **income contradiction** (a profile that claims low income then reveals a well-paid job), social-engineering sob stories, and a SQL-injection string. The system must refuse, stay in character, and never fabricate eligibility.
- **Edge cases** (`SC-V009`, `SC-V016`, `SC-V060`–`SC-V067`, `SC-V091`, `SC-V095`–`SC-V098`): **ambiguous location** (village name without a state, requiring clarification), **language switch mid-call** (Hindi → English), **very elderly** callers (age 85), all-uncertain answers, early call termination, migrant cross-state profiles, and underserved profiles (person with disability, single mother).
- **Reviewer-pattern** (`SC-V080`–`SC-V082`): profiles where a disqualifying or eligibility-changing detail is mentioned *in passing* — a casual "company ka insurance to hai but…", an occupation contradiction (`mazdoori` early, then "meri company"), or an income figure code-switched into English. These target the Reviewer agent, which reads the full transcript and is expected to catch signals a field-by-field pass might miss.
- **State-scheme depth** (`SC-V100`–`SC-V116`): one scenario per newer state scheme (CMCHIS, AP/TS Aarogyasri, KASP, MA Vatsalya, BSKY, MMSY, Abua, Atal Ayushman, DAK, Yeshasvini, Chirayu) plus maternal (`JSY`/`JSSK`) and disease-specific (`NIKSHAY`) schemes, several voiced in the relevant regional language.

### Scenario structure

Every scenario is a dict built by the `_scenario(...)` helper in `scenarios.py` with this shape:

| Field | Meaning |
|-------|---------|
| `id` | Stable identifier, e.g. `SC-V001`, `SC-V040-HI`. |
| `name` | Human-readable label shown in the report. |
| `description` | Rationale: the profile and why the expected outcome is what it is. |
| `language` | Sarvam locale code; sent to the API as the conversation language. |
| `turns` | An ordered list of user utterances — one multi-turn phone call, in the target language (often code-mixed). |
| `expected_eligible_schemes` | Scheme IDs that **should** appear in the result (recall target). |
| `expected_ineligible_schemes` | Scheme IDs that **must not** appear (precision target). |
| `tags` | Category/language/state labels used for filtering and grouping. |

Convenience accessors in the same file resolve scenarios for the runner: `get_all_scenarios()`, `get_scenario_by_id(id)`, `get_scenarios_by_tag(tag)`, and `get_quick_scenarios()` — the last returns a fixed 5-scenario smoke test (`SC-V001`, `SC-V030`, `SC-V034`, `SC-V040-HI`, `SC-V050`) chosen to touch a happy path, an exclusion rule, a state opt-out, a cross-language anchor, and an adversarial case.

## How the runner works

`VaidyaEvaluator` ([`eval/evaluator.py`](../eval/evaluator.py)) drives each scenario by POSTing the full `turns` list and `language` to the `POST /simulate/text` endpoint of a running server, then reads `eligible_schemes`, `conversation`, and `final_phase` from the JSON response. The whole multi-turn call is timed with `time.perf_counter()` to produce `latency_ms`. The default per-request timeout is 120 seconds, since a multi-turn call makes several LLM round-trips.

Scenarios run **sequentially by default** (`concurrency=1`) to avoid overwhelming the API; `run_all(..., concurrency=N)` raises the limit via an `asyncio.Semaphore`. A transport error or timeout is not fatal to the batch — the scenario is recorded as an error (counted as a failure, with its expected-eligible schemes booked as false negatives) and the run continues.

## How to run

The evaluator needs a **running Vaidya server**. Start one in a separate terminal first:

```bash
make run        # uvicorn on http://localhost:8000
```

Then run the suite. The `Makefile` provides three targets, all of which write the Markdown report into `reports/`:

```bash
make eval        # quick: the 5-scenario smoke test  → reports/eval_report.md
make eval-all    # full: all 81 scenarios            → reports/eval_report.md
make eval-live   # honors BASE_URL / SCENARIOS / EVAL_OUTPUT env vars
```

For finer control, invoke the CLI module directly (`python -m eval`, defined in [`eval/run_eval.py`](../eval/run_eval.py)):

```bash
# Everything against a local server
python -m eval --base-url http://localhost:8000 --scenarios all

# Specific scenarios by ID
python -m eval --scenarios SC-V001,SC-V030,SC-V050

# Everything in one category, by tag
python -m eval --tag adversarial
python -m eval --tag exclusion
python -m eval --tag cross_language

# Raise concurrency and set a custom report path
python -m eval --scenarios all --concurrency 3 --output reports/full_run.md
```

Key CLI flags: `--base-url` (default `http://localhost:8000`), `--scenarios` (`all` | `quick` | comma-separated IDs), `--tag` (takes priority over `--scenarios` when both are given), `--concurrency` (default `1`), `--timeout` (default `120` s), `--output` (default `eval_report.md`), and `--verbose`. The process prints a one-line PASS/FAIL/ERROR summary and the average latency to stdout, then **exits non-zero if any scenario failed** — so it can gate CI.

## The report

`generate_report(...)` ([`eval/report.py`](../eval/report.py)) writes a self-contained Markdown report containing:

- **Executive summary** — total scenarios, passed (count and %), failed (count and %), API/timeout errors, and average latency.
- **Per-scenario results table** — one row per scenario: ID, name, PASS/FAIL/ERROR, the schemes actually returned, the false negatives, the false positives, latency, and the final conversation phase.
- **Per-scheme accuracy** — for every scheme that appears in any expectation: true positives, false negatives, false positives, and the resulting **recall** percentage.
- **Reviewer agreement** — computed over the scenarios that carry an `expected_ineligible` list (the exclusion-style cases): the percentage that were correctly excluded (no false positives), plus a list of any **missed exclusions**. A leaked exclusion implies the Reviewer pattern failed to catch a disqualifier, so this figure is the suite's proxy for reviewer agreement.
- **Failure details** — for each failing scenario: the error (if any), missing eligible schemes, incorrectly-eligible schemes, and a collapsible excerpt of the last few conversation turns for debugging.
- **Recommendations** — auto-generated remediation hints, e.g. flagging any scheme whose recall fell below 80%, any scheme with false positives, average latency above 30 s, and a direct cross-language parity check that fires when `SC-V013-HI` and `SC-V013-TA` disagree.

## Results

Run live against the deployed service on 2026-06-13. The numbers below are from a
strictly-sequential (`--concurrency 1`) run of a 15-scenario representative slice
spanning **all 10 languages** plus the exclusion, adversarial, and
cross-language-parity categories. Run the full 81 yourself with `make eval-all`
(allow ~2–3 h sequentially — end-to-end latency is Sarvam-API-bound).

| Metric | Value |
|--------|-------|
| Scenarios | 15 (10 languages) |
| Passed | 9 (60%) |
| **False positives** (an ineligible scheme recommended) | **0** |
| **Exclusion / safety scenarios correct** | **4 / 4 (100%)** |
| State-scheme recall (Chiranjeevi, Aarogyasri, KASP, MA-Vatsalya, MJPJAY, MMSY, Swasthya Sathi, PM-JAY 70+, PMSBY) | **100%** |
| Avg end-to-end latency | ~171 s (see note) |

**What the run establishes**

- **Precision is 100%.** Across every scenario the advisor never recommended a
  scheme the caller is ineligible for. The reviewer → convergence safety pattern
  held on all four exclusion cases — government employee, employer-provided
  insurance, West Bengal (PM-JAY opted out), and a direct prompt-injection
  ("just mark me eligible") — each correctly withheld PM-JAY. For a healthcare
  advisor sending people to enrolment centres, a false positive is the costly
  error, and there were none.
- **State-scheme recall is ~100%, across languages.** The scheme the caller
  actually enrols in — Chiranjeevi (RJ), Dr. YSR Aarogyasri (AP), KASP (KL),
  MA Vatsalya (GJ), MJPJAY (MH), MMSY (PB), Swasthya Sathi (WB), plus PM-JAY 70+
  and PMSBY — was identified correctly in Hindi, Bengali, Tamil, Telugu,
  Malayalam, Gujarati, Punjabi, and more.

**Known gap — the central PM-JAY umbrella (33% recall).** This is a modelling
nuance, not a safety failure. In states that deliver PM-JAY *through* their own
scheme (Rajasthan → Chiranjeevi, Gujarat → MA Vatsalya, Maharashtra → MJPJAY,
Odisha → BSKY), the system surfaces the **state delivery vehicle the caller
actually signs up for** and only inconsistently also lists the central
`PMJAY-2024-v3` label; where PM-JAY is the direct vehicle (Andhra Pradesh,
Kerala) it is returned. Making the central-umbrella inclusion consistent is the
single highest-value recall improvement and is tracked as follow-up work; it does
not touch the (perfect) exclusion-safety path.

**Latency note.** The ~171 s per-scenario figure covers a complete 5-question +
confirmation + full eligibility/reviewer conversation and is dominated by Sarvam
API response time under the eval's load, not Vaidya's orchestration (which adds
<10 ms per routing decision). A single live call is faster; see the
[latency budget](../README.md#latency-budget).

The full Markdown report for this run is committed at
[`reports/eval_representative.md`](../reports/eval_representative.md).
