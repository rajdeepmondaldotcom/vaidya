# Vaidya Development Guide

## What is this?

Vaidya is a voice-first multi-agent healthcare scheme navigator for India. A user calls a phone number, speaks in any of 23 Indian languages, answers 5 questions about their family and income, and hears — in their language — which government healthcare schemes they qualify for.

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
├── agents/          # 5 agents + convergence checker + shared scheme_utils + constants
├── models/          # Pydantic v2 data models (conversation, profile, scheme, api)
├── schemes/data/    # 46 healthcare scheme JSON files (17 central + 29 state)
├── prompts/templates/  # 5 LLM prompt templates (.txt files)
├── knowledge/       # ChromaDB store, loader, embeddings
├── pipeline/        # ConversationManager (turn orchestration), translator, translation_terms
├── compliance/      # PII masking, consent tracking, audit trail
├── session/         # Redis-backed session state
├── voice/           # STT/TTS wrappers, language detection
├── sarvam/          # Async SarvamClient wrapper with circuit breaker resilience
├── telephony/       # Twilio + Pipecat voice pipeline (optional: pip install .[telephony])
├── api/routes/      # FastAPI endpoints (health, conversation, simulate, schemes, compliance, voice)
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
pytest tests/unit/ -v -p no:logfire     # 565+ unit tests, no external deps
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
- `chat()` / `chat_json()` — LLM (sarvam-105b / sarvam-30b, free tier)
- `translate()` — Mayura v1
- `tts()` — Bulbul v3
- `stt()` — Saaras v3

## Adding a New Scheme

1. Create `src/vaidya/schemes/data/new_scheme.json` following the SchemeRecord model
2. Run `python scripts/seed_knowledge.py` to index into ChromaDB
3. Add test scenarios in `eval/scenarios.py`
4. Run eval to verify accuracy

## Current Scope

- 46 schemes (17 central + 29 state) covering all Indian states/UTs
- 23 languages: 11 voice (TTS + STT) + 12 text-only (STT + translate)
- Voice languages: Hindi, Tamil, Bengali, Telugu, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Odia, English
- Text simulation mode + real voice calls via Twilio telephony
- Pipecat voice pipeline: Twilio WebSocket → Sarvam STT → Orchestrator → Sarvam TTS → caller
- ChromaDB for knowledge store
- Redis for session state with atomic pipelines
- Per-session turn locking, circuit breaker resilience, cost tracking
- Railway deployment ready (railway.toml)

## Voice Call Setup (Real Phone Calls)

1. `pip install .[telephony]` -- installs pipecat-ai + twilio
2. Get a Twilio account at twilio.com/try-twilio
3. Run `python scripts/setup_twilio.py --configure --base-url https://your-app.up.railway.app`
4. Set env vars: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`, `VOICE_WEBSOCKET_URL`
5. Deploy to Railway: `railway up` (or any host with WebSocket support)
6. Call the number -- Twilio streams audio to our WebSocket, Pipecat handles STT/TTS

## Deployment

```bash
# Local testing
make run                              # Start server on localhost:8000

# Railway (recommended for production)
railway up                            # Deploy from Dockerfile
railway add redis                     # Add managed Redis
# Set SARVAM_API_KEY, TWILIO_*, VOICE_WEBSOCKET_URL in Railway dashboard

# Docker
docker compose up                     # Local with Redis + ChromaDB
```

## PRD Reference

The complete PRD is at: `/briefings/2026-04-13-vaidya-prd.md` in the j-search-os repo.
