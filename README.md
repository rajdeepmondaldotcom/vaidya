# Vaidya

**Voice-first multi-agent healthcare scheme navigator for 55 crore Indians.**

One phone call. Any Indian language. Vaidya listens, asks 5 questions, and tells you — in your language, in a human voice — which government healthcare schemes you qualify for, what documents you need, and where to go.

No app. No signup. No screen.

---

## The Problem

India runs 50+ government healthcare schemes across central and state levels. PM-JAY alone has 30+ state variants. **55 crore people are eligible. Only 37 crore have enrolled.** The barrier is not policy — it is discovery. The current path requires literacy, internet access, English comprehension, and free time. The target population has none of these.

## How Vaidya Works

```
User calls a phone number
         │
         ▼
┌─────────────────────────┐
│  Saaras V3 (STT)        │  Auto-detects language from speech
│  23 Indian languages     │  Handles code-mixed input
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  ORCHESTRATOR            │  Deterministic state machine
│  7 conversation phases   │  Routes to specialized agents
│  LLM-free routing        │  <10ms per decision
└────────────┬────────────┘
             │
     ┌───────┼───────┬──────────────┐
     │       │       │              │
     ▼       ▼       ▼              ▼
  INTAKE  ELIGIBILITY  REVIEWER   GUIDANCE
  Agent    Agent       Agent      Agent
     │       │          │           │
     │  field-by-field  │   next steps
     │  matching + RAG  │   documents
  5 questions            │   nearest CSC
     │   ┌───────────┐  │
     │   │CONVERGENCE│◄─┘
     │   │  CHECKER   │  Both agents must agree
     │   └─────┬─────┘  Disagreement → conservative + caveat
     │         │
     └─────────┼─────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  Bulbul V3 (TTS)        │  Natural voice in user's language
│  11 Indian languages     │  <3 second voice-to-voice
└─────────────────────────┘
```

## The Reviewer Pattern

The core innovation. At 55 crore beneficiary scale, a 2% false-positive rate = **1.1 crore people sent to CSC centers where they'll be turned away.**

The Eligibility Agent does structured field-by-field matching. The Reviewer Agent independently processes the **full conversation transcript** — catching exclusion criteria mentioned in passing (e.g., employer insurance in a code-mixed aside 3 turns ago). When agents disagree, the system resolves conservatively and logs immutable reasoning traces.

This is what makes Vaidya deployable at government scale, not demo scale.

## Phase 1 Schemes (8)

| Scheme | Coverage | Key Eligibility |
|--------|----------|-----------------|
| PM-JAY | ₹5L/family/year | SECC 2011, income <₹2.5L |
| PM-JAY 70+ | Additional ₹5L | Age 70+, any income |
| Chiranjeevi (Rajasthan) | ₹25L/family/year | NFSA free / ₹850/yr premium |
| Swasthya Sathi (West Bengal) | ₹5L/family | Universal — all WB residents |
| MJPJAY (Maharashtra) | ₹5L/family/year | Ration card holders |
| PMSBY | ₹2L accidental | Age 18-70, bank account, ₹20/yr |
| ESIC | Comprehensive | Salaried <₹21K/month |
| Arogya Karnataka | ₹5L/family/year | NFSA household |

## Tech Stack

```
Runtime:        Python 3.11+ / FastAPI
AI:             Sarvam AI (sarvamai SDK) — zero external AI dependencies
Models:         Sarvam-105B (eligibility/reviewer), Sarvam-30B (intake/guidance)
Knowledge:      ChromaDB (hybrid dense + sparse retrieval)
Session:        Redis (30-min TTL, dropped-call recovery)
Compliance:     PII masking, consent tracking, immutable audit trail
Container:      Docker + docker-compose
```

## Quickstart

### Prerequisites
- Python 3.11+
- Docker (for Redis + ChromaDB)
- [Sarvam AI API key](https://docs.sarvam.ai) (free tier available)

### Setup

```bash
# Clone
git clone https://github.com/rajdeepmondaldotcom/vaidya.git
cd vaidya

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your SARVAM_API_KEY

# Start infrastructure
docker compose up -d redis chromadb

# Run
make run
```

### Try it

```bash
# Text-based conversation simulation
curl -X POST http://localhost:8000/simulate/text \
  -H "Content-Type: application/json" \
  -d '{
    "language": "hi-IN",
    "turns": [
      "Mujhe sarkaari health scheme ke baare mein jaanna hai",
      "Main Rajasthan mein rehta hoon",
      "Ghar mein 5 log hain",
      "Daily mazdoori karta hoon",
      "Nahi, koi insurance nahi hai",
      "Haan, bachche ke liye ilaaj chahiye"
    ]
  }'
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check |
| POST | `/conversation/start` | Start a session |
| POST | `/conversation/{id}/turn` | Send a message |
| GET | `/conversation/{id}` | Get conversation state |
| POST | `/simulate/text` | Full text conversation |
| GET | `/schemes` | List all schemes |
| GET | `/schemes/{id}` | Get scheme details |

## Architecture

### 5-Layer Design

1. **Telephony Gateway** — SIP trunk (Exotel) or WhatsApp (Samvaad)
2. **Voice Pipeline** — Saaras V3 (STT) + Bulbul V3 (TTS), streaming WebSocket
3. **Agent Orchestration** — FastAPI, deterministic state machine, 5 specialized agents
4. **Knowledge Layer** — 8 schemes in ChromaDB, hybrid RAG, state-filtered retrieval
5. **Data & Compliance** — PII masking, consent ledger, immutable audit trail

### 7 Conversation Phases

```
WELCOME → OPEN_ELICITATION → INTAKE (5 Qs) → PROCESSING → RESULTS → GUIDANCE → CLOSURE
                                                  ↑                              │
                                                  └──────── (user has more Qs) ──┘
```

### Latency Budget

| Step | Target | Technique |
|------|--------|-----------|
| Orchestrator routing | <10ms | Pure Python match/case |
| Intake/Guidance agent | 800ms | Sarvam LLM, shorter prompts |
| Eligibility + Reviewer | 1200ms | Parallel execution via asyncio.gather |
| Translation | 400ms | Skip if same language |
| **Total** | **~2.5s** | Under 3s voice-to-voice target |

## Development

```bash
make dev          # Install with dev dependencies
make lint         # Ruff lint + format check
make type-check   # mypy --strict
make test         # Run all tests
make test-unit    # Run unit tests only
```

## Evaluation

```bash
# Run full evaluation suite (requires running server + API key)
python -m eval --base-url http://localhost:8000

# Quick smoke test (3 scenarios)
python -m eval --scenarios quick

# Specific scenarios
python -m eval --scenarios SC-V001,SC-V002,SC-V006
```

## Deployment Phases

| Phase | Scale | Infrastructure |
|-------|-------|---------------|
| **Demo** (now) | 10-50 calls | Single container, Railway/Render |
| **Pilot** | 1,000 calls | K8s cluster, PostgreSQL, all 11 languages |
| **National** | 10,000+ | Chanakya on-premises per state |

## Cost Model

| Component | Per Call (₹) |
|-----------|-------------|
| STT (Saaras V3) | 1.50 |
| LLM (free tier) | 0.00 |
| Translation | 0.40 |
| TTS (Bulbul V3) | 0.45 |
| Telephony | 3.00 |
| **Total** | **~₹5.85** |

## License

MIT

## Author

**Rajdeep Mondal** — Senior Data Scientist, multi-agent systems specialist.
Built production multi-agent systems with 20+ specialized agents curating 260K biomedical records at 93% recall (first-author bioRxiv publication). The reviewer pattern in Vaidya is a direct evolution of the architectural innovation that lifted accuracy from 78% to 93%.
