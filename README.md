# Vaidya

**One phone call. Any Indian language. Find out which government healthcare schemes you qualify for.**

*Vaidya* (वैद्य) is the Sanskrit word for healer. In rural India, the vaidya was the person you went to when you didn't know what was wrong or where to go. Not a specialist. A guide. Someone who listened, understood your situation, and told you what to do next in words you understood.

That's what this system does. Not diagnosis. Not treatment. Discovery. Which government schemes exist for you, what documents you need, and where to go. The name is deliberate: Vaidya is a trusted first point of contact, not an authority. It says "mil sakti hai" (you may be eligible), never "you are eligible." Final verification happens at the Jan Seva Kendra. The system advises. The human decides.

India has 50+ healthcare schemes. PM-JAY alone has 30+ state variants. 55 crore people are eligible. 18 crore haven't enrolled. The barrier isn't policy. It's discovery. The target population doesn't have literacy, internet access, or free time to navigate English PDFs on government websites.

Vaidya solves this with a phone call. The user speaks in their language, answers 5 questions, and hears back which schemes they qualify for, what documents to bring, and where to go. No app. No signup. No screen.

## How it works

A 5-agent system with a deterministic state machine at the center.

```
User speaks → Saaras v3 (STT, 23 languages) → ORCHESTRATOR (state machine, not LLM)
                                                      │
                          ┌───────────────────────────┼──────────────┐
                          ▼                           ▼              ▼
                       INTAKE                    ELIGIBILITY     REVIEWER
                    (5 questions)               (LLM + RAG)   (full transcript)
                          │                           │              │
                          │                           └──────┬───────┘
                          │                          CONVERGENCE CHECK
                          │                       (both must agree to output)
                          └───────────────────────────┬──────────────┘
                                                      ▼
                                                   GUIDANCE
                                              (results + next steps)
                                                      │
                                                      ▼
                                        Bulbul v3 (TTS, 11 languages)
                                                      │
                                                      ▼
                                          User hears the answer
```

The orchestrator is pure Python. No LLM in the routing loop. Match/case state machine with 7 phases. Under 10ms per routing decision.

The agents do the thinking. The orchestrator does the traffic.

## The constraint that shaped the architecture

At 55 crore beneficiary scale, a 2% false-positive rate means 1.1 crore people get sent to a CSC center and turned away. That's the number that drove the reviewer pattern.

The **Eligibility Agent** does structured field-by-field matching against the scheme corpus. The **Reviewer Agent** independently reads the full conversation transcript and catches what the structured matching missed: an employer insurance mention in a code-mixed aside three turns ago, a contradiction between early and late answers, a government job disclosed in passing.

Both agents run in parallel. When they agree, the result goes out. When they disagree, the system resolves conservatively: "mil sakti hai, lekin Jan Seva Kendra mein final confirm hoga."

Every disagreement is logged with both reasoning traces. This is what makes it deployable at government scale, not demo scale.

## Schemes covered

| Scheme | Coverage | Who qualifies |
|--------|----------|---------------|
| PM-JAY | ₹5L/family/year | SECC 2011 families, income below ₹2.5L |
| PM-JAY 70+ | Additional ₹5L | Anyone aged 70+, regardless of income |
| Chiranjeevi (Rajasthan) | ₹25L/family/year | NFSA families free, others ₹850/year |
| Swasthya Sathi (West Bengal) | ₹5L/family | All WB residents, no income criteria |
| MJPJAY (Maharashtra) | ₹5L/family/year | Ration card holders |
| PMSBY | ₹2L accidental | Ages 18-70 with a bank account, ₹20/year |
| ESIC | Comprehensive | Salaried workers under ₹21K/month |
| Arogya Karnataka | ₹5L/family/year | NFSA/BPL households |

61 schemes across central and state programs. Real eligibility rules, real exclusion logic, real enrollment steps. Each stored as a validated JSON file with field-level data. At runtime, Vaidya evaluates every applicable scheme for the caller: central schemes plus the caller's state schemes, or the full registry when the state is unknown.

## What's under the hood

```
Python 3.11 / FastAPI
Sarvam AI: sarvam-105b (eligibility/reviewer), sarvam-30b (intake/guidance)
           saaras:v3 (STT), bulbul:v3 (TTS), mayura:v1 (translation)
ChromaDB for state-filtered scheme retrieval
Redis for session state (30-min TTL, dropped-call recovery)
PII masking (Aadhaar, phone, PAN), consent tracking, immutable audit trail
Docker + docker-compose for local dev
```

Zero dependencies outside the Sarvam SDK and FastAPI ecosystem. No LangChain. No CrewAI. The orchestration is custom because the routing decisions are deterministic and the failure modes are specific to this domain.

## Run it

```bash
git clone https://github.com/rajdeepmondaldotcom/vaidya.git
cd vaidya

pip install -e ".[dev]"
cp .env.example .env        # add your SARVAM_API_KEY
docker compose up -d redis chromadb
python scripts/seed_knowledge.py
make run
```

Then test with a simulated conversation:

```bash
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
      "Bachche ke liye ilaaj chahiye"
    ]
  }'
```

API key: sign up at [dashboard.sarvam.ai](https://dashboard.sarvam.ai). Free tier gives ₹1,000 in credits. The LLM endpoints (sarvam-105b, sarvam-30b) are free. Text simulation mode uses only the LLM. Total cost for the demo: ₹0.

## API

| Method | Path | What it does |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/conversation/start` | Start a session |
| POST | `/conversation/{id}/turn` | Send a user message, get a response |
| GET | `/conversation/{id}` | Current conversation state |
| POST | `/simulate/text` | Full multi-turn text conversation |
| GET | `/schemes` | List all schemes |
| GET | `/schemes/{id}` | Single scheme detail |
| DELETE | `/compliance/data/{phone_hash}` | Delete all user data (DPDP Act) |

## Testing

```bash
make test              # 216 tests, runs in under a second
make lint              # ruff check + format
```

64 evaluation scenarios covering: per-scheme eligibility, exclusion rules, cross-language parity, adversarial inputs (prompt injection, Aadhaar probing), emotional distress handling, and the reviewer-catches-what-eligibility-missed scenario.

```bash
python -m eval --scenarios quick    # 5-scenario smoke test
python -m eval --scenarios all      # full 64-scenario suite
```

## Latency budget

| Step | Target |
|------|--------|
| Orchestrator routing | <10ms (pure Python, no LLM) |
| Intake/Guidance agent | ~800ms (sarvam-30b) |
| Eligibility + Reviewer | ~1200ms (parallel via asyncio.gather) |
| Translation | ~400ms (skipped if same language) |
| **Total per turn** | **Under 3 seconds** |

## Cost per call

The `/costs` report uses actual tracked usage: audio seconds for STT, characters for
translation/TTS/language ID, pages for document intelligence, and call duration for
telephony when `TELEPHONY_RATE_INR_PER_MINUTE` is configured. It also separates usage
by model and mode, so fast-routing (`sarvam-30b`) and regular accuracy routing
(`sarvam-105b`) are visible even though both Sarvam chat models are currently free.
Rates below follow the [Sarvam API pricing docs](https://docs.sarvam.ai/api-reference-docs/pricing).

Formula:

```text
total =
  ceil(stt_audio_seconds) * stt_rate_per_second
  + tts_chars * tts_rate_per_char
  + translate_chars * translate_rate_per_char
  + language_id_chars * language_id_rate_per_char
  + vision_pages * vision_rate_per_page
  + ceil(telephony_seconds / 60) * TELEPHONY_RATE_INR_PER_MINUTE
```

Example: 3-minute voice call, no diarization, 2K translated chars, 1.5K TTS chars,
100 language-ID chars, and optional carrier cost at ₹1/min:

| Component | Rate | Per Call (₹) |
|-----------|------|-------------|
| STT (Saaras v3, transcribe/translate/verbatim/translit/codemix) | ₹30/hour | 1.50 |
| STT with diarization | ₹45/hour | 2.25 |
| LLM fast/regular (`sarvam-30b`/`sarvam-105b`) | Free | 0.00 |
| Translation (Mayura v1 / Sarvam Translate v1) | ₹20/10K chars | 0.40 |
| TTS (Bulbul v3) | ₹30/10K chars | 0.45 |
| Language ID | ₹3.5/10K chars | 0.04 |
| Telephony, if configured at ₹1/min | deployment-specific | 3.00 |
| **Total, standard STT + telephony** | | **~₹5.39** |
| **Total, diarized STT + telephony** | | **~₹6.14** |

At 10,000 calls/day, that's ~₹17.5L/month. At a million calls/day, ~₹17.5 crore/month. The LLM being free is what makes the unit economics work at scale.

## What's next

**Phase 1 (now):** 61 schemes across all states/UTs, 23 languages, text simulation mode, and real voice calls via Twilio.

**Phase 2:** automated scheme-corpus refresh, WhatsApp via Samvaad, CSC integration, and NHA API verification.

**Phase 3 (national):** Chanakya on-premises per state health department. Air-gapped. Full pipeline runs locally. Generalizes beyond healthcare to pensions, agriculture subsidies, education scholarships.

## License

MIT
