# Phase 4 Industry Standards Research

Status: completed 2026-05-06.
Constraint: research refines the Phase 1-3 plan only. It does not redirect the architecture away from FastAPI, Redis, Sarvam, Pipecat, Twilio, or the deterministic orchestrator.

## Sources Reviewed

- Twilio Webhooks Security: https://www.twilio.com/docs/usage/webhooks/webhooks-security
- Twilio TwiML `<Stream>`: https://www.twilio.com/docs/voice/twiml/stream
- Pipecat Speech Input and Turn Detection: https://docs.pipecat.ai/pipecat/learn/speech-input
- Sarvam Streaming Speech-to-Text API: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/streaming-api
- Sarvam Build Voice Agent with Pipecat: https://docs.sarvam.ai/api-reference-docs/integration/build-voice-agent-with-pipecat
- Sarvam Saaras model page: https://docs.sarvam.ai/api-reference-docs/getting-started/models/saaras
- Sarvam Text-to-Speech overview: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/overview
- OWASP Logging Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- Google Conversation Design error handling: https://developers.google.com/assistant/conversation-design/errors
- Microsoft voice agent prompt best practices preview: https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/voice-agents-prompt-best-practices
- NIST Privacy Framework crosswalk to India's DPDP Act and Rules: https://www.nist.gov/privacy-framework/nist-privacy-framework-10-digital-personal-data-protection-act-2023-and-rules-2025

## Findings Against The Plan

### Twilio Webhook Security

Twilio says applications should verify that Twilio sent the webhook before responding, using the `X-Twilio-Signature`, the exact webhook URL, all request parameters, and the account auth token. Twilio also warns that parameters can change and recommends the SDK validator instead of a custom implementation.

Impact on plan:

- Confirms the plan to replace the placeholder signature presence check with `RequestValidator`.
- Add a deployment note: reverse proxies must preserve the public scheme/host or validation will fail because Twilio signs the exact URL it called.
- Keep form params flexible; do not hardcode only today's Twilio fields.

### Twilio Media Streams

Twilio documents `<Connect><Stream>` as bidirectional, with the call continuing only while the WebSocket remains open. `<Stream>` supports `statusCallback` and `statusCallbackMethod`. Twilio also supports nested `<Parameter>` values, which are passed to the WebSocket server in the Start message, with a combined `name` + `value` length limit under 500 characters.

Impact on plan:

- Confirms passing `phone_hash` as a custom Stream parameter.
- Refines the plan to wire `voice_status_callback_url` into the `<Stream>` tag when configured.
- Confirms the terminal silence handler should end the task/call; bidirectional streams cannot be stopped with `<Stop>`.

### Pipecat Turn Detection

Pipecat documents that VAD emits raw `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame`, and turn strategies derive higher-level `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame`. It also notes lower `start_secs` is more responsive but can trigger on brief sounds, while defaults should be adjusted only after profiling.

Impact on plan:

- Confirms our current frame-level handling is in the right place: `VaidyaAgentProcessor` should react to `UserStartedSpeakingFrame` and `TranscriptionFrame`.
- Does not require adding local Silero VAD to transport params.
- Keep silence thresholds conservative and testable; do not over-tune without call recordings/metrics.

### Sarvam STT/TTS

Sarvam Streaming STT docs recommend Saaras v3, document modes including `transcribe`, `translate`, `verbatim`, `translit`, and `codemix`, and document `high_vad_sensitivity`, `vad_signals`, and 8 kHz telephony sample-rate matching. Sarvam's Saaras page says language codes can be omitted or set to unknown for auto-detection and that the response includes `language_code` and probability. Sarvam TTS docs position Bulbul v3 as supporting streaming audio for interactive voice agents.

Impact on plan:

- Confirms using `SarvamSTTSettings(model="saaras:v3", language=None, vad_signals=True, high_vad_sensitivity=True)` and sample rate 8000.
- Confirms the language auto-switcher should trust STT language metadata when present, but keep a fallback when it is absent.
- Keep Bulbul v3 and streaming Pipecat service; no redesign.

### Voice UX Error Handling

Google's conversation design guidance says no-input handling should assume the user may not have heard the question, repeat or rephrase concisely, provide more support on the second prompt, and gracefully exit after repeated no-input/no-match events. Microsoft guidance for voice agents emphasizes deterministic structure, short turn-taking, one-question-at-a-time prompts, hard safety boundaries, and standardized closing behavior.

Impact on plan:

- Confirms the 6s nudge, 12s contextual reprompt, 20s closure shape.
- Refines copy: avoid "I didn't hear you" framing; use "I'm listening" and then re-ask.
- Supports keeping the state machine and one-question intake structure.

### Privacy And Logging

OWASP logging guidance says sensitive personal data and PII, including health data, government identifiers, vulnerable people, and telephone numbers, should usually be removed, masked, sanitized, hashed, or encrypted before logging. NIST's 2026 crosswalk notes India's DPDP Act is consent-based for digital personal data processing.

Impact on plan:

- Confirms hashing caller identifiers before logs and Stream custom params.
- Confirms not committing `.env.production`.
- Confirms audit logs should retain event shape without raw caller phone numbers.

## Research-Driven Plan Refinements Only

1. Add actual Twilio signature validation and document proxy URL correctness.
2. Pass only hashed caller identity through Twilio custom parameters.
3. Add optional `<Stream statusCallback>` wiring using existing `voice_status_callback_url`.
4. Use Sarvam service-side VAD signals instead of local Silero transport params.
5. Keep two no-input recovery prompts plus graceful exit; do not add more retries.
6. Keep all architecture decisions from Phase 1-3.
