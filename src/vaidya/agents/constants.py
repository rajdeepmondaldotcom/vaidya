"""Named constants for the agent layer.

Centralises magic numbers that were previously scattered across
orchestrator.py, convergence.py, eligibility.py, and intake.py.
"""

# Silence thresholds (seconds) -- PRD Section 3.2
SILENCE_REASSURE = 5
SILENCE_REPHRASE = 10
SILENCE_CONNECTION_LOSS = 15
SILENCE_END_CALL: float = 20.0

# Scheme processing
MAX_SCHEMES_PER_LLM_CALL = 20
RAG_TOP_K = 10

# Confidence penalties applied during convergence
SINGLE_AGENT_CONFIDENCE_PENALTY = 0.8
DISAGREEMENT_CONFIDENCE_PENALTY = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.7

# Intake
MAX_INTAKE_QUESTIONS = 5
