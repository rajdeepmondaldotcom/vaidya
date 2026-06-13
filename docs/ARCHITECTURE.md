# Vaidya Architecture

Vaidya is a voice-first, multi-agent healthcare-scheme navigator for India. A
caller dials a phone number, speaks in any of **23 Indian languages**, answers
five short questions about their household and income, and hears — in their own
language — which of **61 government healthcare schemes** they qualify for.

This document is the technical deep-dive: how the conversation is driven, how
the agents collaborate, how the Sarvam model stack is wired, and the design
tradeoffs behind each choice. It is meant to be read alongside the source —
every claim here maps to code under `src/vaidya/`.

---

## 1. Overview & data flow

The system is a **5-agent pipeline driven by a deterministic state-machine
orchestrator**. The orchestrator is plain Python, not an LLM — it decides
*which* agent runs next; the agents (and a pure-Python convergence checker) do
the work.

```
User speech → STT → ORCHESTRATOR (state machine, NOT LLM) → AGENTS → TTS → User hears response
                         │
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
      INTAKE         ELIGIBILITY     REVIEWER       GUIDANCE
    (5 questions)   (LLM + RAG)   (transcript)   (next steps)
         │               │               │
         │               └───────┬───────┘
         │                       ▼
         │              CONVERGENCE CHECK
         │              (pure Python, no LLM)
         └───────────────────────┘
```

The flow, end to end:

1. **STT** — Sarvam **Saaras v3** transcribes the caller's speech (code-mix
   mode on voice calls, so Hinglish-style mixing survives transcription).
2. **Orchestrator** — a deterministic state machine (`agents/orchestrator.py`)
   routes the turn by conversation phase. No LLM is consulted for routing on
   the happy path.
3. **Agents** — depending on phase, the orchestrator dispatches the Intake,
   Eligibility, Reviewer, or Guidance agent. Eligibility and Reviewer run
   **in parallel**, then a pure-Python **Convergence** check merges their
   verdicts.
4. **TTS** — Sarvam **Bulbul v3** synthesizes the spoken reply, which is
   streamed back to the caller.

Two surfaces share this exact pipeline: a **text simulation** channel (the
HTTP conversation API, used for evals and local testing) and **real phone
calls** over Twilio + Pipecat. The `channel` argument (`"text"` vs `"voice"`)
threads through the orchestrator so voice calls get a phone-friendly opener
(`Orchestrator.handle_turn`), but the agent logic is identical on both.

The internal processing language for the agents is `en-IN`
(`ConversationManager._AGENT_LANG`). Inbound and outbound translation hops sit
at the edges of the pipeline — with deliberate exceptions on the intake path
(see [§6](#6-per-turn-data-flow)).

---

## 2. The deterministic state-machine orchestrator

The orchestrator's defining decision: **routing is a `match`/`case` over seven
conversation phases, not an LLM call.** This is the first of the project's core
design decisions, and it shapes everything downstream.

The seven phases (`models/conversation.py :: ConversationPhase`, routed in
`Orchestrator._route_by_phase`):

| Phase | Responsibility |
|---|---|
| `WELCOME` | Language selection + consent/disclaimer handshake |
| `OPEN_ELICITATION` | Listen to a free-form opening statement |
| `INTAKE` | The five structured questions |
| `PROCESSING` | Run Eligibility + Reviewer + Convergence |
| `RESULTS` | Deliver all eligible schemes in one turn |
| `GUIDANCE` | Answer follow-ups; next steps, documents, CSC directions |
| `CLOSURE` | Farewell / restart |

```python
match context.phase:
    case ConversationPhase.WELCOME:
        return await self._handle_welcome(...)
    case ConversationPhase.INTAKE:
        return await self._handle_intake(...)
    case ConversationPhase.PROCESSING:
        return await self._handle_processing(...)
    ...
    case _:
        # Only here — an unrecognized phase — do we fall back to the LLM.
        return await self._llm_fallback(context, user_input)
```

**Why a state machine and not an LLM router?**

- **Determinism.** The same input in the same phase always routes the same
  way. There is no prompt to drift, no temperature, no token budget to blow.
  In a healthcare context, "where does this turn go?" must never be a coin
  flip.
- **Latency.** A `match`/`case` plus a few dict lookups is sub-millisecond.
  An LLM router would add a full network round-trip to *every* turn just to
  decide routing — before any useful work happens.
- **Auditability.** Each phase transition is explicit in code and logged, so
  the whole conversation graph can be reasoned about and tested without
  mocking a model.

**The LLM is a fallback, not the driver.** It is invoked in exactly two
narrow situations (`Orchestrator._llm_fallback`, model `sarvam-30b`):

1. Input that does not fit the current phase at all (the `case _:` arm).
2. Mid-flow intent changes and repairs — the caller asks a side question
   during intake, says "repeat that," "start over," or "I'm done." These are
   classified before phase routing in `_pre_route_repair`, which leans on a
   lightweight heuristic intent classifier (`agents/turn_intent.py`) and only
   escalates to the LLM for genuinely ambiguous cases.

Everything else — the entire happy path — is pure Python.

The orchestrator also owns conversational-resilience behaviors that need no
model: **silence escalation** (a tiered nudge → reprompt → graceful hang-up
schedule from `agents/constants.py`), **repeat escalation**
(rephrase → simplify → offer an SMS), and **emotional-distress fast-tracking**
(skip remaining intake and go straight to results with an empathetic, slowed
TTS profile).

---

## 3. The five agents

Each agent has one job and a narrow interface. Four are LLM-backed; the fifth
(Convergence) is pure Python. The orchestrator wires them together; the agents
never call each other directly.

### Intake (`agents/intake.py`)
Collects the five-question profile: (1) state/district, (2) family size,
(3) income & occupation, (4) existing coverage, (5) health need. The question
order is deliberate — an easy location question first to build rapport, the
sensitive income question third (after some trust), the optional health-need
question last.

Intake is **LLM-first** by default, running on `sarvam-30b` natively in the
caller's language: the model reads (say) Tamil directly and extracts canonical
English field values, so no inbound translation is needed. It accumulates the
profile across turns, runs a confirmation recap at the end, and is biased
toward *proceeding* (a false "not confirmed" would trap the caller in a
correction loop, whereas the downstream results are already conservative).
A deterministic heuristic **fast path** exists but is **opt-in and off by
default** (see [§7](#7-key-design-decisions--tradeoffs)).

### Eligibility (`agents/eligibility.py`)
Does **structured, field-by-field matching**. For each candidate scheme the
LLM (`sarvam-105b`) checks the structured `UserProfile` against the scheme's
criteria — income thresholds, occupation include/exclude lists, geographic
restrictions, exclusion rules, age and family criteria — and returns a verdict
(`ELIGIBLE` / `INELIGIBLE` / `UNCERTAIN`) with confidence, matched/failed
criteria, and a reasoning trace.

To keep LLM work flat as the corpus grows, the agent first **filters schemes by
the caller's state**, then (when a `KnowledgeStore` is present) uses **RAG
retrieval to rank and prune** the applicable set to a top-k. The retrieval
ranking is cached per session against a profile fingerprint, so unchanged
profiles reuse the prior ordering instead of re-querying the vector store.
Candidates are evaluated in **small, bounded, concurrent batches** that share a
single rendered system prompt.

### Reviewer (`agents/reviewer.py`)
Independently re-derives eligibility **from the raw conversation transcript**,
*not* from the structured profile. This is the deliberate asymmetry that makes
the safety pattern work. The Reviewer exists specifically to catch what
field-by-field matching misses:

- exclusion criteria mentioned in passing ("company ka insurance to hai");
- corrections the caller made mid-conversation;
- code-mixed asides and contradictions between early and late statements.

Its reasoning path is **transcript-evidence-based**, and it surfaces fields the
structured profile missed. Like Eligibility, it filters by state and evaluates
in concurrent batches, but it reads the narrative rather than the form.

### Guidance (`agents/guidance.py`)
Turns the converged eligible list into **one TTS-ready spoken message**
(`sarvam-30b`): an intro with the count, one concise advisory line per scheme
(name + benefit), and a closing offer of fuller detail plus an SMS of the full
list. Uncertain/conservative matches are framed honestly ("you *may* qualify —
confirm at the Jan Seva Kendra"). There is no per-scheme drip-feed; the caller
hears every scheme at once and can then ask about any of them in the `GUIDANCE`
phase. The agent has deterministic fallbacks so the caller always hears their
schemes even if the LLM returns nothing usable.

### Convergence checker (`agents/convergence.py`)
**Pure Python, no LLM.** The decision matrix that merges the Eligibility and
Reviewer verdicts — described in detail next.

---

## 4. The Reviewer → Convergence safety pattern

This is the **core safety mechanism** of the system, and the reason there are
two eligibility-style agents instead of one.

**The shape of it:**

- The **Eligibility** agent matches the *structured profile* field-by-field.
- The **Reviewer** agent independently reads the *full transcript*.
- The **Convergence** checker (`ConvergenceChecker.check`) compares the two
  verdicts for every evaluated scheme. **Both must agree before a scheme is
  confidently surfaced as eligible.**

The two agents run **in parallel** (`Orchestrator._execute_agents` creates both
`asyncio` tasks and waits on them together), so the second opinion costs almost
no extra wall-clock time.

Convergence sorts every scheme into one of four buckets:

| Outcome | Result |
|---|---|
| Both agree **eligible** | Surface as eligible (higher of the two confidences) |
| Both agree **ineligible** | Drop (recorded in `agreed_ineligible`) |
| Both **uncertain** | Surface conservatively as "you may qualify" |
| **Disagree** | Resolve against the transcript (see below) |

**Resolving disagreements** (`_resolve_disagreement`): the checker identifies
the divergent field (from the two agents' `failed_criteria`, falling back to
keyword matching over their reasoning traces), then checks whether the
**transcript** contains evidence about that field:

- **Evidence found** → trust the **Reviewer's** verdict (it is the agent that
  reads transcript evidence), with a logged caveat.
- **No clear evidence** → fall back to `UNCERTAIN` with a "verify at the Jan
  Seva Kendra" caveat.

Crucially, the resolution is **conservative in the direction that protects the
caller**. An unresolved disagreement becomes `UNCERTAIN` and is *still
surfaced* as a "you may qualify, please confirm" result — **unless** one agent
found an explicit disqualifier, in which case the scheme is dropped. Single-
agent and disagreement-derived matches also carry **confidence penalties**
(`SINGLE_AGENT_CONFIDENCE_PENALTY = 0.8`,
`DISAGREEMENT_CONFIDENCE_PENALTY = 0.7`).

**Why this matters in healthcare:**

- **It catches exclusions mentioned in passing.** A caller might mention an
  employer insurance policy as an aside while answering a different question.
  Field-by-field matching can miss it; the transcript-reading Reviewer is built
  to catch exactly that, and Convergence then resolves the disagreement against
  the transcript.
- **It avoids both failure modes.** A false *positive* (telling someone they
  qualify when they do not) erodes trust and sends people to a Jan Seva Kendra
  for nothing. A false *negative* (turning away a genuinely-eligible caller
  with "no scheme matched") is worse — they needed the help most. The two-agent
  agreement requirement guards the positive case; the conservative
  "surface-as-uncertain" rule guards the negative case.

Every disagreement is recorded with both reasoning traces, so the compliance
audit trail can reconstruct exactly why a scheme was surfaced, dropped, or
flagged uncertain.

The pattern degrades gracefully: if the Reviewer is unavailable, Convergence
builds a single-agent result from Eligibility alone, with confidence penalties
and a `reviewer_unavailable` flag (`Orchestrator._build_single_agent_convergence`).

---

## 5. Model routing across the Sarvam stack

Every AI call goes through one async wrapper, `sarvam/client.py`, which exposes
`chat()` / `chat_json()` (LLM), `translate()`, `tts()`, and `stt()`. Models are
configured in `config.py` and named in `sarvam/models.py`.

| Stage | Model | Where |
|---|---|---|
| STT | **Saaras v3** (`saaras:v3`) | All inbound speech; `codemix` mode on voice |
| Orchestrator fallback | **sarvam-30b** | `orchestrator_model` |
| Intake | **sarvam-30b** | `intake_model` |
| Eligibility | **sarvam-105b** | `eligibility_model` |
| Reviewer | **sarvam-105b** | `reviewer_model` |
| Guidance | **sarvam-30b** | `guidance_model` |
| TTS | **Bulbul v3** (`bulbul:v3`) | All outbound speech |
| Translation (voice) | **Mayura v1** (`mayura:v1`) | 11 voice languages, colloquial |
| Translation (text) | **sarvam-translate v1** | All 23 languages, formal |

**The routing principle is a latency-vs-accuracy split.** Both LLMs are on
Sarvam's free tier, so the choice is purely about behavior, not cost:

- **`sarvam-30b` for the fast, conversational agents** — orchestrator fallback,
  intake, guidance. These turns happen face-to-face with the caller and must
  feel instant; `sarvam-30b` is roughly **~3× faster** than the flagship. These
  agents also run **natively multilingual** — they read and reply in the
  caller's language directly, which is what lets the intake path skip
  translation round-trips entirely.
- **`sarvam-105b` for the accuracy-critical agents** — Eligibility and Reviewer.
  Getting a benefits verdict wrong has real human cost, so these get the
  flagship 105B model. They run off the caller's critical path (during the
  `PROCESSING` phase, behind a spoken "let me check that" filler), so the extra
  latency is hidden.

Three implementation details make this work on the free tier:

1. **`reasoning_effort = "low"` everywhere** (eligibility, reviewer, intake,
   guidance). `sarvam-30b`/`105b` are always-on reasoning models whose
   reasoning shares the same 4096-token output budget as the answer. Verbose
   reasoning starves the content channel and truncates the JSON to empty — the
   cause of multi-minute "no result" failures. `"low"` is the floor that keeps
   both conversational *and* scheme-eval turns from truncating.

2. **Small, highly-parallel scheme batches.** Eligibility/Reviewer evaluate the
   candidate corpus in batches of **3 schemes** (`scheme_eval_batch_size`) with
   up to **8 parallel batches** (`scheme_eval_max_parallel_batches`). This is a
   *latency* optimization, not just a token-budget one: many small `105b` calls
   run concurrently and finish in roughly one call's time, whereas one big
   batch is a single long call that also risks the per-call timeout and a retry.
   Measured on the paid tier, `batch_size=10` blew a 6-turn simulation past
   220s; `batch_size=3` keeps it near real-time.

3. **Tiered timeouts.** Conversational `sarvam-30b` turns use a tighter
   `conversational_llm_timeout_seconds` (30s) so a single hung call on a simple
   turn can't stall the caller, while the slower eligibility/reviewer tail keeps
   the longer `llm_timeout_seconds` (45s).

**STT, TTS, and translation** sit at the pipeline edges. Saaras v3 covers all
23 languages. Bulbul v3 synthesizes the 11 voice languages (each with a
distinct speaker). Translation is **model-routed by language**
(`get_translate_model`): Mayura v1's colloquial style for the 11 voice
languages, sarvam-translate v1's formal style for the full 23 — the 12
text-only languages reach the caller over SMS/WhatsApp rather than voice.

---

## 6. Per-turn data flow

`ConversationManager` (`pipeline/conversation.py`) is the turn-orchestration
layer between the HTTP/voice edge and the `Orchestrator`. A single turn:

1. **Per-session lock.** Acquire an `asyncio.Lock` keyed by `call_id` so two
   concurrent inputs for the same call can't interleave and corrupt state.
2. **Load session** from Redis (`session_ttl_seconds = 1800`). A missing
   session returns a localized "session expired" message.
3. **PII masking.** The raw utterance is masked before it ever touches the
   audit log (`compliance/pii.py`).
4. **Inbound translation (conditional).** Normally the caller's text is
   translated to the internal `en-IN`. But on **language-selection** and
   **intake-bound** turns the raw utterance is passed straight through — the
   natively-multilingual intake agent reads the caller's language directly,
   so the inbound translate hop would be pure redundant latency
   (`_should_skip_inbound_translation`).
5. **Orchestrate.** The orchestrator runs the phase handler, dispatching agents
   as needed.
6. **Outbound translation (conditional).** The agent reply is translated back
   to the caller's language *unless* the agent marked it `already_localized`
   (intake and guidance do, since they already speak the caller's language).
7. **Persist & audit.** Updated context is written back to Redis; the turn and
   any eligibility decision are written to the audit trail; per-call cost is
   recorded.

### Resilience layer

The pipeline is built to keep a caller on the line even when individual
services wobble.

- **Circuit breakers** (`sarvam/resilience.py`). Per-service breakers (STT,
  TTS, LLM, translate) trip after a failure threshold and fail-fast for a
  recovery window, then probe in half-open state — so a cascade in one Sarvam
  API doesn't drag down calls to the others. Retries with backoff live in the
  client (`_retry_async`).
- **Speculative eligibility** (`EligibilityAgent.start_speculative`). The moment
  the profile is complete enough to evaluate (by the last intake question), the
  orchestrator kicks off a **non-blocking** background eligibility pass. When
  the conversation reaches `PROCESSING`, that result is reused **only if a
  full-profile fingerprint still matches** — so results are usually ready the
  instant intake ends, with byte-identical output to a fresh pass. It is purely
  a latency optimization: any fingerprint mismatch, failure, or absent entry
  falls back to a synchronous compute, so correctness is identical to never
  having speculated. The speculative map is bounded and cancelled on session
  end so background tasks never leak.
- **Graceful degradation** (`pipeline/degradation.py`). A `DegradationManager`
  tracks consecutive per-service failures and computes an ordered level:
  `FULL → NO_REVIEWER → REDUCED_LANGUAGES → SCRIPTED → SMS_ONLY`. The
  orchestrator checks this **before** dispatching agents — e.g. at
  `NO_REVIEWER` it proactively skips the Reviewer task rather than waiting for
  it to time out, and Convergence falls back to the single-agent path.
- **Caching.**
  - *TTS* (`sarvam/tts_cache.py`): fixed/templated prompts — the greeting, the
    five intake questions, the processing filler, silence nudges, the closure
    line — are byte-identical every call, so their synthesized audio is cached
    (LRU, keyed on everything that affects the waveform). A second render is a
    dict lookup, not a network call. Failures are never cached.
  - *Translation* (`pipeline/translator.py`): a bounded per-instance LRU cache
    memoizes identical translations, keyed on
    `(text, source, target, speaker_gender, output_script)`.
  - *Retrieval*: the RAG scheme ranking is cached per session against a profile
    fingerprint (above).
- **Cost tracking** (`sarvam/cost.py`). Every API call records INR cost by
  service/model/mode. The LLMs are free-tier (₹0); STT, TTS, and translation
  are billed per second/character. Per-call totals are stamped onto the session
  (`session_cost_inr`) and threshold alerts fire in the logs. This makes the
  unit economics of a call observable in real time.

---

## 7. Key design decisions & tradeoffs

**State machine vs LLM router.** Routing is deterministic Python
(`match`/`case` over 7 phases), with the LLM reserved for genuinely
unexpected input. The tradeoff is explicitness: every conversational path must
be coded and tested rather than emergent from a prompt. In return we get
sub-millisecond, fully-deterministic, auditable routing — the right call when
the downstream output is a benefits eligibility decision.

**Two-agent reviewer safety vs single-agent simplicity.** Running both a
structured Eligibility agent and a transcript-reading Reviewer, then requiring
their **convergence**, costs a second `105b` evaluation. We pay it because the
agents fail in *different* ways — field matching misses passing asides; a
transcript read can miss a structured threshold — and requiring agreement
catches each other's blind spots. The parallel execution hides most of the
latency; the conservative resolution rules ensure a disagreement never silently
turns a caller away.

**LLM-first intake with an opt-in heuristic fast path.** Intake defaults to
LLM-first on `sarvam-30b` (`intake_fast_path_enabled = False`). A deterministic
heuristic fast path — which skips the LLM for short, unambiguous answers — is
available but **off by default**. The reasoning: LLM-first is robust across the
huge range of phrasings and languages a caller might use and showcases native
multilingual extraction, whereas brittle keyword heuristics should never be the
*default* path to correctness. Operators who hit high volume can flip the flag
(`INTAKE_FAST_PATH_ENABLED=true`) to trade a little robustness for lower latency
and cost, with the end-of-intake confirmation step catching any heuristic
mis-read.

**JSON scheme data → ChromaDB, with an in-memory fallback.** The 61 schemes
live as **human-readable, git-trackable JSON files** (`schemes/data/`,
following the `SchemeRecord` model), indexed into ChromaDB at startup for
vector retrieval that scales as the corpus grows. The tradeoff — a vector store
dependency — is softened by a graceful fallback: when no store is available (or
retrieval yields nothing usable), the agents evaluate the **full
state-applicable set** instead. Retrieval can only ever *prune* to a top-k; it
can never silently drop a genuinely-applicable scheme, because the no-store path
keeps the complete list.

**Minimal dependencies.** Orchestration is custom — no LangChain, no CrewAI.
The runtime leans on `sarvamai` + FastAPI, with Redis for session state and
ChromaDB for the knowledge store. Custom orchestration is what makes the
deterministic state machine, the speculative-eligibility optimization, and the
convergence safety pattern possible without fighting a framework's control flow.

---

## Source map

| Concern | File |
|---|---|
| State-machine routing | `src/vaidya/agents/orchestrator.py` |
| Reviewer pattern / safety | `src/vaidya/agents/convergence.py` |
| Field-by-field matching + RAG + speculation | `src/vaidya/agents/eligibility.py` |
| Transcript-based independent review | `src/vaidya/agents/reviewer.py` |
| Five-question profile elicitation | `src/vaidya/agents/intake.py` |
| Spoken results + SMS | `src/vaidya/agents/guidance.py` |
| Turn orchestration, translation, audit | `src/vaidya/pipeline/conversation.py` |
| Graceful degradation levels | `src/vaidya/pipeline/degradation.py` |
| Model routing & timeouts | `src/vaidya/config.py` |
| Model names & language support | `src/vaidya/sarvam/models.py`, `src/vaidya/voice/language.py` |
| Circuit breakers / retries | `src/vaidya/sarvam/resilience.py`, `src/vaidya/sarvam/client.py` |
| Per-call cost tracking | `src/vaidya/sarvam/cost.py` |
| TTS / translation caches | `src/vaidya/sarvam/tts_cache.py`, `src/vaidya/pipeline/translator.py` |
| Scheme data | `src/vaidya/schemes/data/*.json` |
