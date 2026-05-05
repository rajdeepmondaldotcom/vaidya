# Phase 8 Post-Implementation Audit

Scope: audit of commits `101941e` and `78fbf69` against the Phase 6 final consolidated plan.

## Implemented

1. Voice route security and privacy

- `POST /voice/incoming` now hashes caller identifiers before logging or TwiML emission.
- TwiML Stream params carry only `phone_hash`, not the raw caller number.
- TwiML attributes are escaped before interpolation.
- Twilio HTTP signature validation uses `twilio.request_validator.RequestValidator` when `twilio_auth_token` is configured.
- Optional Stream `statusCallback` and `statusCallbackMethod="POST"` are wired when `voice_status_callback_url` is configured.

2. Voice session creation

- `WS /voice/stream` validates the Twilio WebSocket signature before accept when configured.
- The route accepts the socket, parses the Twilio/Pipecat handshake, derives `phone_hash`, then starts the conversation.
- `run_voice_pipeline()` accepts parsed `transport_type` and `call_data`, and parses only as a fallback.
- Session cleanup only runs after `call_id` exists.

3. Pipecat/Sarvam compatibility

- Removed unsupported local `vad_analyzer` transport wiring.
- Sarvam STT is configured with 8 kHz telephony audio, `vad_signals=True`, and `high_vad_sensitivity=True`.
- Twilio serialization uses 8 kHz sample rate and only enables auto-hangup when Twilio credentials and call SID are present.
- TTS language updates now use typed `SarvamTTSSettings` deltas rather than deprecated dict updates.

4. Voice UX behavior

- `VaidyaAgentProcessor` starts silence watching after bot speech, cancels on caller speech/transcription, escalates at 6/12/20 seconds, and emits `EndTaskFrame` on terminal silence.
- First-transcription language detection now uses central language normalization, including Pipecat enum values and Odia `or-IN` -> `od-IN`.
- Detected language switches persist into the session and update downstream TTS voice/language.
- Idle watcher cleanup wakes and cancels pending tasks during processor teardown.

5. Conversation consistency and copy

- `POST /conversation/start` now passes the requested `channel` into `ConversationManager.start_conversation()`.
- Welcome silence reprompt covers all 11 supported voice languages.
- Intake prompt no longer asks for "one short word" while giving multi-word acknowledgement examples, and explicitly says not to echo PII.

6. Runtime and test stability

- Telephony dependencies are installed in the Docker build via the telephony extra.
- `.env.production` is ignored so local production secrets are not committed.
- `Settings` ignores unrelated deployment env vars, which fixes app startup under Railway-style environments.
- KnowledgeStore unit tests use a deterministic fake Chroma client rather than depending on local SQLite FTS extension support.

## Verification

- `uv run --extra dev --extra telephony ruff check src/ tests/ eval/` passed.
- `uv run --extra dev --extra telephony ruff format --check src/ tests/ eval/` passed.
- `uv run --extra dev --extra telephony pytest tests/unit tests/integration -q -p no:logfire` passed: 679 tests, 1 third-party `audioop` deprecation warning.

## Findings For Phase 9

1. `POST /voice/status` is now wired from TwiML but does not validate Twilio signatures.

- Risk: unauthenticated clients can write lifecycle-looking events to logs.
- Required fix: reuse Twilio HTTP signature validation for status callbacks when `twilio_auth_token` is configured.

2. `POST /voice/status` logs raw `CallSid`.

- Risk: call identifiers are not phone numbers, but they are still provider-side identifiers and should be treated as operational metadata rather than raw log fields.
- Required fix: log a hashed call identifier and keep only non-sensitive status/duration fields.

3. Twilio validation dependency failure is indistinguishable from a bad signature.

- Risk: if `twilio_auth_token` is configured without the Twilio package installed, `/voice/incoming` returns `403` instead of surfacing deployment misconfiguration.
- Required fix: return `503` for validator-unavailable cases and keep `403` for invalid signatures.

4. `voice_stream()` docstring still states the old order of operations.

- Risk: low; implementation is correct, but the docstring contradicts the final plan.
- Required fix: update the docstring to say handshake parsing happens before conversation creation.

## Deviations

- No Terraform commands were run.
- No GitHub push was performed.
- No frontend work was added because this repo is backend/API/voice only.
- Real field validation remains outside local scope: call recordings, latency traces, Twilio proxy URL behavior, and Sarvam production API behavior still need staging/production observation.
