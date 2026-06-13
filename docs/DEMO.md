# Vaidya Demo Walkthrough

A two-minute, no-phone-needed way to see Vaidya end to end. You start the API
locally, then drive a simulated phone call from the terminal using
`scripts/simulate_call.py`. The simulator speaks to the same
`/conversation/start` and `/conversation/{id}/turn` endpoints a real Twilio call
uses, so what you see in the transcript is exactly what a caller would hear
(minus the audio).

This page also lists three curated, copy-pasteable conversations across three
languages that are known to work end to end.

## Prerequisites

- Python environment set up (`make dev`).
- `SARVAM_API_KEY` exported -- the eligibility and guidance agents call Sarvam
  for the LLM + RAG step. Without it the server still starts, but the intake/
  eligibility turns will not produce real scheme matches.
- Redis is optional for a local demo; the server falls back to in-memory
  session state when Redis is not configured.

## Run a demo

In one terminal, start the API:

```bash
make run            # serves http://localhost:8000
```

In a second terminal, start the call simulator:

```bash
python scripts/simulate_call.py
```

You will see a header, the welcome message, and a `Caller:` prompt. Type each
turn and press Enter. After every turn the simulator prints:

- the **Vaidya** reply, and
- a dim metadata line: `phase`, `schemes_found`, per-turn `latency`, and running
  `cost` in rupees.

When the intake completes (5 answers: location, family size, income,
occupation, existing coverage), Vaidya runs eligibility + the reviewer pass and
reads back the schemes you qualify for. Type `/quit` to end the call and print a
tidy summary: caller turns, total Sarvam cost in rupees, and the numbered list
of eligible schemes.

### Useful flags and in-call commands

```bash
python scripts/simulate_call.py --language bn-IN     # start in Bengali
python scripts/simulate_call.py --language en-IN     # start in English
python scripts/simulate_call.py --base-url http://localhost:8000
```

- `/status` -- print the current phase, captured profile, and eligible schemes.
- `/lang <code>` -- switch the language code for subsequent turns (e.g.
  `/lang ta-IN`).
- `/quit` -- end the call and print the summary.

> Tip for screen recordings: keep the terminal window wide enough that the
> Vaidya replies do not wrap mid-sentence. The speaker labels (`Caller` /
> `Vaidya`) and the per-turn metadata line are designed to read cleanly on a
> recording.

## Curated scenarios

Each scenario below is a full intake. Start the simulator in the listed language
and paste the turns one at a time, waiting for each Vaidya reply before sending
the next. The "Expected result" notes which schemes should appear in the final
summary. (Exact wording of replies varies, but the eligible schemes are stable.)

Phrasings are transliterations (Latin script) -- the same style a real caller's
speech-to-text produces -- so you can type them on any keyboard.

### 1. Hindi -- Rajasthan daily-wage worker with a heart condition

```bash
python scripts/simulate_call.py --language hi-IN
```

Turns:

1. `Namaste, mujhe apne parivaar ke liye sarkari yojana ke baare mein jaanna hai`
2. `Ji haan, main Rajasthan mein rehta hoon, Jaipur ke paas ek gaon mein`
3. `Meri family mein 5 log hain - main, meri patni, do bacche aur meri maa`
4. `Main daily wage pe kaam karta hoon, construction mein. Mahine ka lagbhag 6-7 hazaar kamata hoon`
5. `Nahi, humare paas koi bhi health insurance nahi hai`
6. `Haan, BPL card hai hamare paas. Aur mujhe dil ki bimari hai, heart ka operation karwana hai`

**Expected result:** eligible for **Pradhan Mantri Jan Arogya Yojana (Ayushman
Bharat)** (central) and **Mukhyamantri Chiranjeevi Swasthya Bima Yojana**
(Rajasthan). The heart-surgery mention is exactly the kind of need PM-JAY's
cardiac packages cover.

### 2. Bengali -- West Bengal farmer with TB

```bash
python scripts/simulate_call.py --language bn-IN
```

Turns:

1. `Namaskar, ami jantte chhai sarkari swasthya yojana aache ki na`
2. `Ami West Bengal e thaki, Hooghly district er ekta gram e`
3. `Amader paribare 5 jon -- ami, amar stree, duti baccha ar amar ma`
4. `Ami chashi, nijer jomite chash kori. Bochore pray 60-70 hajar aay`
5. `Na, kono health insurance nei amader`
6. `Hyan, ration card aache. Ar amar TB dhora poreche, jokkha rog, chikitsa lagbe`

**Expected result:** eligible for **Swasthya Sathi (West Bengal Health Scheme)**
(state, universal -- no income test) and the **National Tuberculosis Elimination
Programme (NTEP)** (central, free TB diagnosis and treatment). Note PM-JAY is
*not* expected here -- West Bengal opted out of PM-JAY, and the reviewer pass
enforces that exclusion.

### 3. English -- Kerala, private job, cataract surgery

```bash
python scripts/simulate_call.py --language en-IN
```

Turns:

1. `Hello, I want to know about government health schemes for my family`
2. `I live in Kerala, in a town near Thrissur`
3. `There are 4 of us -- me, my wife and two children`
4. `I work a private job at a small shop, I earn about 2 lakh a year`
5. `No, we do not have any health insurance`
6. `We have a ration card. My father needs cataract surgery for his eyes`

**Expected result:** eligible for the **National Programme for Control of
Blindness and Visual Impairment (NPCBVI)** (central -- free cataract surgery and
eye care) and **Karunya Arogya Suraksha Padhathi (Kerala)** (the state's
AB-PMJAY umbrella scheme). The cataract mention is what surfaces NPCBVI via the
condition-aware retrieval step.

## What you are seeing under the hood

The orchestrator is a deterministic state machine (not an LLM) that walks the
call through welcome -> intake -> eligibility -> review -> guidance. The
Eligibility agent does field-by-field matching against the scheme catalog; the
Reviewer agent re-reads the full transcript to catch exclusions mentioned in
passing (for example, employer insurance, or a state that opted out of PM-JAY).
Both must agree before a scheme is reported. That convergence step is why the
expected results above are stable run to run.
