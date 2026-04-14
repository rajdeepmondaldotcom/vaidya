# Vaidya Development Guide

## What is this?

Vaidya is a voice-first multi-agent healthcare scheme navigator for India. A user calls a phone number, speaks in Hindi/Tamil/Bengali, answers 5 questions about their family and income, and hears — in their language — which government healthcare schemes they qualify for.

## Architecture

5-agent system with a deterministic state machine orchestrator:

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

## Key Design Decisions

1. **Orchestrator is a state machine, not an LLM.** 7 conversation phases as Python `match/case`. LLM only for unexpected input. This gives deterministic routing and <10ms per decision.

2. **Reviewer pattern is the core safety mechanism.** The Eligibility Agent does field-by-field matching. The Reviewer Agent reads the FULL transcript and catches exclusion criteria mentioned in passing. Both must agree before output.

3. **Zero dependencies beyond sarvamai + FastAPI.** No LangChain, no CrewAI. Custom orchestration matches Sarvam's Arya philosophy.

4. **Scheme data as JSON files → ChromaDB at startup.** Human-readable, git-trackable, reviewer-friendly. Vector retrieval for scalability.

## Project Structure

```
src/vaidya/
├── agents/          # 5 agents + convergence checker + base protocol
├── models/          # Pydantic v2 data models (conversation, profile, scheme, api)
├── schemes/data/    # 8 healthcare scheme JSON files
├── prompts/templates/  # 5 LLM prompt templates (.txt files)
├── knowledge/       # ChromaDB store, loader, embeddings
├── pipeline/        # ConversationManager (turn orchestration), translator
├── compliance/      # PII masking, consent tracking, audit trail
├── session/         # Redis-backed session state
├── voice/           # STT/TTS wrappers, language detection
├── sarvam/          # Async SarvamClient wrapper
├── api/routes/      # FastAPI endpoints (health, conversation, simulate, schemes, compliance)
├── app.py           # FastAPI factory with lifespan
└── config.py        # Pydantic BaseSettings
```

## Commands

```bash
make dev          # Install with dev deps
make lint         # ruff check + format
make test         # pytest (unit tests)
make run          # Start FastAPI server
make docker       # docker compose up
```

## Testing

```bash
pytest tests/unit/ -v -p no:logfire     # 161+ unit tests, no external deps
pytest tests/integration/ -v            # Integration tests with mocked LLM
python -m eval --scenarios quick        # 5-scenario smoke test (needs running server)
python -m eval --scenarios all          # Full 64-scenario eval suite
```

## Key Files to Read First

1. `src/vaidya/agents/orchestrator.py` — The state machine brain. Start here.
2. `src/vaidya/agents/convergence.py` — The safety mechanism (reviewer pattern).
3. `src/vaidya/prompts/templates/` — All agent prompts. Quality here = output quality.
4. `src/vaidya/schemes/data/pmjay.json` — Example scheme data structure.
5. `src/vaidya/models/scheme.py` — Core data models.

## Sarvam API Usage

All AI calls go through `src/vaidya/sarvam/client.py`:
- `chat()` / `chat_json()` — LLM (sarvam-m4, free tier)
- `translate()` — Mayura v1
- `tts()` — Bulbul v2
- `stt()` — Saarika v2 (Phase 2, placeholder in Phase 1)

## Adding a New Scheme

1. Create `src/vaidya/schemes/data/new_scheme.json` following the SchemeRecord model
2. Run `python scripts/seed_knowledge.py` to index into ChromaDB
3. Add test scenarios in `eval/scenarios.py`
4. Run eval to verify accuracy

## Phase 1 Scope

- 8 schemes (5 central + 3 state)
- 3 languages (Hindi, Tamil, Bengali)
- Text simulation mode (no telephony)
- ChromaDB for knowledge store
- Redis for session state

## PRD Reference

The complete PRD is at: `/briefings/2026-04-13-vaidya-prd.md` in the j-search-os repo.
