# Vaidya

[![CI](https://github.com/rajdeepmondaldotcom/vaidya/actions/workflows/ci.yml/badge.svg)](https://github.com/rajdeepmondaldotcom/vaidya/actions/workflows/ci.yml)

**Call one number, speak any Indian language, answer five questions, and find out which government health schemes you qualify for — what to bring, and where to go.**

*Vaidya* (वैद्य) is the old word for the village healer — the person you went to when you didn't know what was wrong or where to go. Not a specialist. A guide who listened, understood your situation, and told you the next step in words you understood. That's the job here. Not diagnosis. Not treatment. Discovery.

## Two real calls — and how to hear it live

The two recordings in [`demo/`](demo/) are the whole thing, end to end — real calls to the live line, trimmed only for dead air:

- [`demo/hindi-vaidya-call.wav`](demo/hindi-vaidya-call.wav) — a caller answering in Hindi
- [`demo/bengali-vaidya-call.wav`](demo/bengali-vaidya-call.wav) — a caller answering in Bengali

You can also dial it yourself, with one honest caveat. The line runs on a Twilio trial account (the why is below), and a trial line only connects calls from numbers verified on the account. So **+1 775 372 2354** won't pick up for a number I haven't verified, and verifying one takes me about two minutes. The recordings are the full experience for anyone; if you'd like to hear it live, tell me the number you'll call from and I'll open the line to it. Either way it answers in Hindi and switches to your language the moment you speak.

## The problem is discovery, not policy

India already has the schemes. PM-JAY alone covers ₹5 lakh per family per year, with more than thirty state variants layered on top. Around 55 crore people are eligible. Roughly 18 crore have never enrolled.

The barrier isn't the policy. It's finding out the scheme exists and that it's meant for you. The people who need it most are the least able to go looking — they don't have the literacy, the internet, or a free afternoon to decode an English PDF on a government portal. So the benefit sits there, real and unclaimed.

Vaidya takes the screen out of the way. You make a phone call, in your language, and you hang up knowing what you can get and where to go for it. No app, no signup, no reading.

The name is a promise about scope. Vaidya is a first point of contact, not an authority. It says *mil sakti hai* — you may be eligible — never *you are eligible*. The final word belongs to the Jan Seva Kendra. The system advises; the human decides.

## How it works

One spoken answer travels a long way before the caller hears a reply. It is transcribed, routed, reasoned over by up to four agents, cross-checked for safety, translated if needed, and spoken back — and almost all of that runs on the Sarvam stack.

```
Caller speaks ─▶ Saaras v3 (STT, 23 languages)
                      │
                      ▼
              ORCHESTRATOR  ── deterministic state machine, not an LLM
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
   INTAKE         ELIGIBILITY        REVIEWER
 (5 questions)   (LLM + RAG)     (reads the full
      │          field matching    transcript)
      │               └───────┬────────┘
      │                       ▼
      │               CONVERGENCE CHECK
      │            (both must agree to speak)
      └───────────────────────┬────────────────┐
                               ▼
                           GUIDANCE  ── schemes + next steps
                               │
                               ▼
                    Bulbul v3 (TTS, 11 voice languages) ─▶ Caller hears the answer
```

**The part I'm most deliberate about is the orchestrator, and it isn't an LLM.** It's plain Python — a `match`/`case` state machine over seven conversation phases — and it makes every routing decision in under ten milliseconds. A language model is unpredictable by design; the control flow of a system that tells people what healthcare they qualify for cannot be. So the orchestrator owns the routing, deterministically, and the agents do their thinking inside the lanes it sets. The model only gets involved for the parts that genuinely need judgment.

**Intake** asks the five questions one at a time and turns spoken answers into structured fields, even when the caller volunteers things out of order or mixes languages mid-sentence.

**Eligibility** matches that profile field by field against the scheme corpus, using retrieval over a vector store so it only reasons about the schemes that could plausibly apply.

**Reviewer** does something different, and at scale more important: it ignores the structured fields and reads the entire raw transcript again, looking for what field-matching tends to miss — an employer-insurance mention dropped in a code-mixed aside three turns ago, a government job disclosed in passing, an answer that quietly contradicts an earlier one.

**Why two checks instead of one?** At 55 crore people, a two percent false-positive rate sends 1.1 crore of them to a centre to be turned away. That single number is the reason eligibility is decided twice, two different ways, and only spoken when both agree. When they disagree, the system resolves conservatively — *you may qualify, confirm at the Jan Seva Kendra* — and logs both reasoning traces. That's the difference between something that demos well and something you could put in front of 55 crore people.

**The Sarvam stack does the heavy lifting, routed by what each step needs.** Saaras hears the caller and Bulbul answers them. The slow, careful work — eligibility and the reviewer — runs on `sarvam-105b` for accuracy. The fast, conversational work — intake and guidance — runs on `sarvam-30b` for speed. Mayura translates when the caller's language and the reasoning language differ. Nothing is bolted on; each model is doing the job it's best at.

**It feels quick because it cheats honestly on latency.** The moment intake has gathered enough, eligibility and the reviewer start running in the background while the caller is still talking, so the answer is usually ready the instant the last question is done. Language is handled the way the caller actually experiences it: Vaidya opens in Hindi, reads the *script* of the first answer to detect the real language, and switches its own voice to match — Bengali, Tamil, and the rest — instead of trusting a language tag that is often wrong on short, name-heavy speech.

There is no LangChain and no CrewAI here. The orchestration is hand-written, because the routing is deterministic and the failure modes are specific to this domain, and I wanted to own both.

## How it's deployed

This runs as a live phone line, not a notebook.

**Railway** runs the FastAPI app, Redis for session state, and ChromaDB for the scheme vectors, all from a Dockerfile with the embedding model baked into the image so a cold start never waits on a download. Sessions live in Redis with a short TTL, so a caller whose line cuts mid-intake can ring back and pick up where they left off.

**Twilio** carries the actual call. When someone dials, Twilio hits a webhook on the app, gets back TwiML that opens a bidirectional Media Stream, and streams the live audio to a WebSocket. From there a Pipecat pipeline wires that stream to Saaras for speech-in and Bulbul for speech-out, with the orchestrator and agents sitting in the middle of the loop. The caller just hears a conversation.

**A note on the number, since you'll notice it's American.** The right way to run this in India is a local number from a provider like Exotel, or Twilio's India route. Both need a registered business and KYC I don't have as an individual, and the India path carries regulatory paperwork I can't clear on my own yet. So I did the next best thing: I built the entire telephony path on Twilio's free trial and deployed it end to end. The trial leaves three fingerprints, and all three are the trial's doing, not the system's — it's a US number, it plays a short "trial account" message before Vaidya answers, and it only connects calls from numbers I've verified on the account. Everything after that message — the live audio, the language switch, the reasoning, the spoken answer — is real and running in production. A paid account clears all three at once: an Indian number, no preamble, open to any caller. Nothing else in the system changes.

## The schemes

| Scheme | Cover | Who qualifies |
|--------|-------|---------------|
| PM-JAY | ₹5L / family / year | SECC-2011 families, income under ₹2.5L |
| PM-JAY 70+ | extra ₹5L | anyone aged 70 or above, any income |
| Chiranjeevi (Rajasthan) | ₹25L / family / year | NFSA families free, others ₹850/year |
| Swasthya Sathi (West Bengal) | ₹5L / family | every WB resident, no income test |
| MJPJAY (Maharashtra) | ₹5L / family / year | ration-card holders |
| PMSBY | ₹2L accidental | ages 18–70 with a bank account, ₹20/year |
| ESIC | comprehensive | salaried workers under ₹21K / month |

That's a sample. The full corpus is 61 schemes — 29 central and 32 state — covering every state and union territory. Each one is a validated JSON file with its real eligibility rules, exclusions, and enrollment steps, kept human-readable and git-tracked so a domain reviewer can check it without reading code. At runtime Vaidya evaluates every scheme that could apply to the caller: the central set plus their state's, or the whole registry when the state isn't known yet.

## What's under the hood

Python 3.11 and FastAPI. The Sarvam SDK for `sarvam-105b` and `sarvam-30b`, Saaras v3, Bulbul v3, and Mayura v1. ChromaDB for state-filtered retrieval, Redis for sessions. Aadhaar, phone, and PAN are masked before anything is stored; consent is tracked; the audit trail is append-only; and one DELETE endpoint wipes a caller's data for DPDP-Act requests. Close to a thousand unit and integration tests run on every push, alongside `ruff` and `mypy --strict`, behind a coverage floor. On top of those, an 81-scenario evaluation suite scores the things that actually matter end to end: per-scheme accuracy, exclusion logic, identical results for the same profile across languages, adversarial inputs like prompt injection and Aadhaar probing, and the case where the reviewer catches what eligibility missed. The methodology is in [docs/EVALUATION.md](docs/EVALUATION.md), the design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and a guided walkthrough in [docs/DEMO.md](docs/DEMO.md).

The economics are the quiet point. A three-minute call costs roughly ₹5 — speech in, speech out, a little translation, and the carrier minute. Both Sarvam chat models are free today, and that single line item is what keeps the unit cost flat at a million calls a day, which is the scale this problem actually lives at.

## Run it

```bash
git clone https://github.com/rajdeepmondaldotcom/vaidya.git && cd vaidya
pip install -e ".[dev]"
cp .env.example .env          # add your SARVAM_API_KEY
docker compose up -d redis chromadb
python scripts/seed_knowledge.py
make run
```

Then hold a full conversation in text, no phone needed:

```bash
curl -X POST http://localhost:8000/simulate/text \
  -H "Content-Type: application/json" \
  -d '{"language":"hi-IN","turns":[
    "Mujhe sarkaari health scheme ke baare mein jaanna hai",
    "Main Rajasthan mein rehta hoon",
    "Ghar mein 5 log hain",
    "Daily mazdoori karta hoon",
    "Nahi, koi insurance nahi hai",
    "Bachche ke liye ilaaj chahiye"]}'
```

A Sarvam key is free at [dashboard.sarvam.ai](https://dashboard.sarvam.ai), the LLM endpoints cost nothing, and text mode uses only the LLM — so the conversation above runs for ₹0.

## What's next

Right now: 61 schemes across every state and UT, 23 languages, text simulation, and real voice calls over Twilio. Next: an automated refresh of the scheme corpus, WhatsApp through Samvaad, and verification against the NHA API. After that, a per-state deployment that runs the whole pipeline locally and air-gapped for health departments that need it — and the same shape generalizes past healthcare, to pensions, agriculture subsidies, and scholarships, which all share the same discovery problem.

## License

MIT
