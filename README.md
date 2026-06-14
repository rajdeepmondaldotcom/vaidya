# Vaidya

[![CI](https://github.com/rajdeepmondaldotcom/vaidya/actions/workflows/ci.yml/badge.svg)](https://github.com/rajdeepmondaldotcom/vaidya/actions/workflows/ci.yml)

**One phone call, in any Indian language, tells you which government health schemes you can get.**

Vaidya (वैद्य) is the Sanskrit word for healer. In a village, the vaidya was who you went to when you didn't know what was wrong or where to turn. Not a specialist. A guide who listened, understood your situation, and told you what to do next in words you understood.

This system does that one thing. Not diagnosis, not treatment. Discovery. Which schemes exist for you, what papers to carry, and where to go. It says you *may* be eligible, never that you *are*. The final word belongs to the person at the Jan Seva Kendra. Vaidya only points the way.

## The problem isn't policy. It's discovery.

India runs more than fifty public health schemes. PM-JAY alone has over thirty state variants. Around 55 crore people are eligible, and roughly 18 crore have never enrolled. The people who need it most can't read an English PDF on a government portal, can't spare a smartphone, and can't lose an afternoon finding out.

So Vaidya meets them on the one device they already have, used the way they already use it. They call, speak their language, answer five questions, and hear which schemes they qualify for, what to bring, and where to go. No app, no signup, no screen.

## Call it

It's live. Dial **+1 775 372 2354** and talk to it in Hindi or Bengali, the way you'd talk to a person — your state, who lives with you, how the household earns, whether anyone has insurance, what you need. It greets you in Hindi, switches to your language the moment you answer, and a minute later reads back your schemes.

## How it works

Five agents, with a plain Python state machine in the middle.

```
User speaks → Saaras v3 (STT, 23 languages) → ORCHESTRATOR (state machine, not an LLM)
                                                     │
                          ┌──────────────────────────┼───────────────┐
                          ▼                          ▼               ▼
                       INTAKE                   ELIGIBILITY       REVIEWER
                    (5 questions)              (LLM + RAG)    (full transcript)
                          │                          │               │
                          │                          └──────┬────────┘
                          │                         CONVERGENCE CHECK
                          │                      (both must agree to answer)
                          └──────────────────────────┬───────────────┘
                                                     ▼
                                                  GUIDANCE
                                            (schemes + next steps)
                                                     │
                                                     ▼
                                          Bulbul v3 (TTS, 11 languages)
```

The orchestrator is not an LLM. It's a match/case state machine that decides the next step in under ten milliseconds. The agents do the thinking; the orchestrator directs traffic. Putting an LLM in the routing loop would make every turn slower and less predictable for no gain, because the routing here is deterministic.

## The decision that shaped everything

At 55 crore people, a two percent false-positive rate sends 1.1 crore of them to a center to be turned away. That number is why two agents check eligibility, not one.

The Eligibility Agent matches the caller's profile against each scheme's rules, field by field. The Reviewer Agent reads the whole conversation again and looks for what structured matching misses: an employer's insurance mentioned in passing, a late answer that contradicts an early one, a government job let slip while naming the family. They run in parallel. When they agree, the answer goes out. When they don't, Vaidya stays careful — you may be eligible, confirm at the Jan Seva Kendra. Every disagreement is logged with both agents' reasoning. That is the line between something you can demo and something a state can run.

## The schemes

| Scheme | Coverage | Who qualifies |
|--------|----------|---------------|
| PM-JAY | ₹5L / family / year | SECC 2011 families, income below ₹2.5L |
| PM-JAY 70+ | Additional ₹5L | Anyone 70 or older, regardless of income |
| Chiranjeevi (Rajasthan) | ₹25L / family / year | NFSA families free, others ₹850/year |
| Swasthya Sathi (West Bengal) | ₹5L / family | All WB residents, no income test |
| MJPJAY (Maharashtra) | ₹5L / family / year | Ration card holders |
| PMSBY | ₹2L accidental | Ages 18–70 with a bank account, ₹20/year |
| ESIC | Comprehensive | Salaried workers under ₹21K/month |
| Arogya Karnataka | ₹5L / family / year | NFSA / BPL households |

Sixty-one schemes, central and state, each a checked JSON file with real eligibility rules, exclusions, and enrollment steps. For every caller, Vaidya weighs each scheme that could apply — the central ones plus their state's, or the whole set until the state is known.

## Built on Sarvam

Saaras v3 turns speech into text across 23 languages. Bulbul v3 speaks the answer back in 11. sarvam-105b runs eligibility and review, sarvam-30b runs intake and guidance, and Mayura v1 translates between the caller's language and the engine's. ChromaDB retrieves schemes filtered by the caller's state, and Redis holds the session so a dropped call can pick up where it left off.

One detail is worth calling out, because it's where voice gets hard. The bot opens in Hindi and switches to the caller's language from their first answer. It decides that language from the script the words come back in, not from the speech model's language tag — a short, name-heavy first reply is exactly what gets mistagged, and a caller speaking Bengali should never be answered in English.

There is no LangChain and no CrewAI. The orchestration is a few hundred lines of Python, because the routing is deterministic and the failure modes belong to this problem, not a framework.

## On latency, honestly

Vaidya's own routing is effectively free. The wait a caller feels is the model calls — speech in, two reasoning passes, speech out — and that time is set by the Sarvam API and how loaded it is. So eligibility runs in the background while the caller is still answering, and the line stays warm with a short spoken update during the search. The call never goes silent.

## What it costs to run

A three-minute voice call costs about ₹5, almost all of it speech-to-text and telephony, because the reasoning models are free on Sarvam today. At a million calls a day that is roughly ₹17 crore a month. The free LLM is what makes the math work at national scale. The `/costs` endpoint reports actual tracked usage per model and mode, following the [Sarvam pricing docs](https://docs.sarvam.ai/api-reference-docs/pricing).

## Run it

```bash
git clone https://github.com/rajdeepmondaldotcom/vaidya.git
cd vaidya
pip install -e ".[dev]"
cp .env.example .env          # add your SARVAM_API_KEY
docker compose up -d redis chromadb
python scripts/seed_knowledge.py
make run
```

Then try a full conversation in text, no phone needed:

```bash
curl -X POST http://localhost:8000/simulate/text \
  -H "Content-Type: application/json" \
  -d '{"language":"hi-IN","turns":[
    "Mujhe sarkaari health scheme ke baare mein jaanna hai",
    "Main Rajasthan mein rehta hoon",
    "Ghar mein paanch log hain",
    "Daily mazdoori karta hoon",
    "Nahi, koi insurance nahi hai",
    "Bas yeh jaanna hai ki kya kya mil sakta hai"]}'
```

Sign up at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) for an API key. The reasoning endpoints are free, so the text demo costs nothing.

## How it's tested

`make check` runs the full gate locally: `ruff`, `mypy --strict`, and 950+ unit and integration tests with a coverage floor. The same gate runs on every push.

On top of that, an 81-scenario evaluation suite scores the thing that actually matters — end-to-end eligibility. It checks per-scheme accuracy and exclusion rules, cross-language parity (the same profile in Hindi, Tamil, and Bengali must return the same schemes), adversarial inputs like prompt injection and Aadhaar probing, distress handling, and the case where the reviewer has to catch what eligibility missed.

```bash
python -m eval --scenarios quick    # 5-scenario smoke test
python -m eval --scenarios all      # the full 81-scenario suite
```

More detail lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/DEMO.md](docs/DEMO.md).

## What's real today

Sixty-one schemes across every state and union territory. Twenty-three languages. Text simulation and real voice calls over Twilio. Deployed, and callable at the number above.

What comes next is breadth, not a rewrite: an automated refresh of the scheme corpus, WhatsApp as a second channel, and verification against the NHA API. The same shape — call, understand, point the way — extends past health to pensions, farm subsidies, and scholarships.

## License

MIT
