"""Named constants for the agent layer.

Centralises magic numbers that were previously scattered across
orchestrator.py, convergence.py, eligibility.py, and intake.py.
"""

# Silence thresholds (seconds) -- PRD Section 3.2
SILENCE_REASSURE = 5
SILENCE_REPHRASE = 10
SILENCE_CONNECTION_LOSS = 15
SILENCE_END_CALL: float = 20.0

# Voice-edge silence escalation (in real calls via Pipecat).
# Tuned shorter than the simulation thresholds above: on a phone call,
# dead air over ~6s already feels broken, and 20s is the terminal cut.
# Each entry is (threshold_seconds, i18n_key). The last entry's
# `is_terminal` flag is True -- after speaking it the call hangs up.
SILENCE_STEPS: list[tuple[float, str, bool]] = [
    (6.0, "silence_nudge", False),
    (12.0, "silence_reprompt_prefix", False),
    (20.0, "silence_closure", True),
]

# Patient silence escalation used after the caller explicitly asks for time.
PATIENT_SILENCE_STEPS: list[tuple[float, str, bool]] = [
    (12.0, "silence_nudge", False),
    (24.0, "silence_reprompt_prefix", False),
    (40.0, "silence_closure", True),
]

# Scheme processing. MAX_SCHEMES_PER_LLM_CALL is a per-call batch size, not a
# corpus-wide cap; callers must batch when evaluating larger candidate sets.
MAX_SCHEMES_PER_LLM_CALL = 20
MAX_PARALLEL_SCHEME_BATCHES = 3
RAG_TOP_K = 10

# Confidence penalties applied during convergence
SINGLE_AGENT_CONFIDENCE_PENALTY = 0.8
DISAGREEMENT_CONFIDENCE_PENALTY = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.7

# Intake
MAX_INTAKE_QUESTIONS = 5
