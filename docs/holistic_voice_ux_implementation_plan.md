# Holistic Voice UX Implementation Plan

Status: Phase 3 revision, pre-research.
Anchor: the current Vaidya architecture remains the source of truth: FastAPI routes, Redis-backed `ConversationManager`, deterministic `Orchestrator`, Sarvam STT/TTS/LLM services, Pipecat telephony transport, Twilio webhook and WebSocket routes.

## Phase 1 Initial Plan

### System Reading

The app is a voice-first healthcare scheme navigator with a clear backend-only architecture:

- `src/vaidya/app.py` owns service construction and dependency wiring.
- `src/vaidya/api/routes/conversation.py` exposes text/web session APIs.
- `src/vaidya/api/routes/voice.py` exposes Twilio webhook, Twilio Media Stream WebSocket, and status callbacks.
- `src/vaidya/telephony/pipeline.py` owns Pipecat transport, serializer, Sarvam STT/TTS, and pipeline runtime.
- `src/vaidya/telephony/agent_processor.py` bridges Pipecat frames to `ConversationManager`, including voice silence handling and first-utterance language switching.
- `src/vaidya/pipeline/conversation.py` owns session lifecycle, translation, audit logging, silence utterance generation, and language persistence.
- `src/vaidya/agents/orchestrator.py` owns the state machine and keeps welcome, intake, processing, results, guidance, and closure deterministic.
- `src/vaidya/i18n/strings/*.json` owns localized spoken copy.

The branch already includes a partial voice UX implementation: language-first onboarding, 6/12/20 second voice silence steps, TTS language switching, Twilio serializer updates, and focused tests. The plan should finish and stabilize that direction rather than redesign it.

### Implementation Goals

1. Make real Twilio voice sessions isolated, recoverable, and privacy-preserving.
2. Make Twilio webhook security real, not a placeholder.
3. Make the installed Pipecat 1.1 voice pipeline start reliably with Sarvam STT/TTS.
4. Make language-first onboarding consistent across voice and text entry points.
5. Make silence, reprompting, and hang-up behavior predictable and phone-friendly.
6. Strengthen tests around the production failure points, not only unit-level helpers.
7. Keep deployment safe: no Terraform commands, no pushes, no secret commits.

### Workstreams

#### 1. Twilio Identity And Session Isolation

Problem:

- `voice_stream()` currently creates `phone_hash` from the constant `b"twilio-call"`.
- Every caller can collide into the same dropped-call recovery index.
- Caller PII is passed as a TwiML custom parameter and logged as `caller`.
- `run_voice_pipeline()` parses the Twilio handshake after the session has already started, so the route cannot use the Twilio call SID or custom parameters for session identity.

Plan:

- In `incoming_call()`, hash the Twilio `From`/`Caller` value immediately.
- Pass only `phone_hash` as the Twilio Stream custom parameter.
- Avoid logging raw caller identifiers.
- Parse the Twilio WebSocket handshake before creating the conversation session.
- Use `customParameters.phone_hash` when present.
- Fall back to a deterministic hash of Twilio `callSid`, then `streamSid`, and finally a process-local unknown marker only if Twilio omits both.
- Pass the already parsed handshake into `run_voice_pipeline()` so the pipeline does not consume the handshake twice.

Edge cases:

- Missing custom parameters.
- Empty `CallSid`.
- WebSocket disconnects before handshake.
- Non-Twilio transport type returned by Pipecat helper.
- Dropped-call recovery should work for the same hashed caller but not merge unrelated callers.

#### 2. Twilio Request Validation

Problem:

- The current webhook checks only whether `X-Twilio-Signature` is present.
- A malicious client can forge a POST with any non-empty signature.

Plan:

- Use Twilio's `RequestValidator` when `twilio_auth_token` is configured.
- Validate the full request URL and form parameters.
- Return `403` on missing or invalid signature.
- Return a service-unavailable response if validation is configured but the `twilio` package is not installed.
- Keep local development usable when no Twilio token is configured.

Edge cases:

- Reverse-proxy URL mismatch can cause false negatives. Document deployment requirement to preserve original scheme/host.
- Form data is already consumed for Twilio params; reuse the parsed form rather than reading twice.

#### 3. Pipecat 1.1 Pipeline Compatibility

Problem:

- `FastAPIWebsocketParams` in installed Pipecat 1.1 does not accept `vad_analyzer`.
- Local Silero VAD detection is therefore wired to a parameter that will raise at runtime.
- Sarvam STT already supports service-side `vad_signals` and broadcasts `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame`.

Plan:

- Remove the unsupported `vad_analyzer` argument from transport params.
- Enable Sarvam STT `vad_signals=True` and `high_vad_sensitivity=True` through `SarvamSTTSettings`.
- Keep `UserStartedSpeakingFrame` cancellation logic in `VaidyaAgentProcessor`.
- Keep `BotStoppedSpeakingFrame` as the idle-watch trigger.
- Keep `PipelineParams()` compatible with Pipecat 1.1.

Edge cases:

- If Sarvam does not emit VAD events, the agent processor still cancels silence on final `TranscriptionFrame`.
- If VAD events are emitted, silence cancellation and interruption happen sooner.

#### 4. Conversation Entry Consistency

Problem:

- `StartConversationRequest.channel` is validated but not passed into `ConversationManager.start_conversation()`.
- HTTP starts can ask for voice but receive the text welcome.

Plan:

- Pass `request.channel` from the conversation route into `start_conversation()`.
- Add tests proving voice starts get the voice welcome path and text starts get the text menu path.

#### 5. Voice Copy And Silence UX

Problem:

- `silence_welcome_reprompt` lists only Hindi, Tamil, and English even though the app supports 11 voice languages.
- The intake system prompt asks for "one short word" but examples include multi-word acknowledgements.
- Welcome and silence copy should be brief enough for phone use and should not echo PII.

Plan:

- Update welcome-reprompt copy to reflect all 11 supported voice languages in a concise phrasing.
- Tighten the intake prompt wording so acknowledgements are short, never PII-echoing, and not contradictory.
- Keep current 6/12/20 second silence thresholds unless research shows a small refinement is needed.

#### 6. Lifecycle And Cleanup

Problem:

- If the WebSocket handshake fails before a call ID exists, the route should not try to end a nonexistent conversation.
- Idle watcher tasks should not survive processor cleanup.

Plan:

- Make `call_id` optional in `voice_stream()` until the session is created.
- End the session only when creation succeeded.
- Add cleanup logic to `VaidyaAgentProcessor` to cancel/wake the idle task.

#### 7. Verification

Run focused and broad checks:

- `uv run --extra dev --extra telephony pytest tests/unit/test_conversation_manager.py tests/unit/test_silence_handler.py tests/integration/test_voice_agent_processor.py -q -p no:logfire`
- Add route/pipeline unit tests for Twilio identity, request validation, and channel pass-through.
- `uv run --extra dev --extra telephony pytest tests/unit tests/integration -q -p no:logfire`
- `uv run --extra dev --extra telephony ruff check src tests eval`

### Non-Goals

- No Terraform commands.
- No push to GitHub.
- No architecture rewrite away from FastAPI, Redis, Sarvam, Pipecat, or the deterministic orchestrator.
- No scheme corpus redesign.
- No frontend work unless the existing repo contains one; this repo is backend/API/voice only.

## Phase 3 Round 1 Revision

The Phase 2 self-audit found four blockers and three refinements. The plan above is revised as follows:

- Treat Twilio handshake parsing as a route responsibility for voice calls, then pass the parsed `transport_type` and `call_data` into `run_voice_pipeline()`.
- Keep `run_voice_pipeline()` backward-compatible by allowing it to parse the handshake when no parsed context is supplied.
- Replace local Silero VAD wiring with Sarvam service-side VAD signals because installed Pipecat 1.1 does not support `vad_analyzer` on `FastAPIWebsocketParams`.
- Add tests that instantiate the pipeline dependency surface enough to catch unsupported Pipecat parameters.
- Make `.env.production` explicitly out of commit scope; it exists locally and may contain secrets.
- Do not commit `.venv` created by `uv`; `.gitignore` already covers virtualenv-style directories.

The implementation direction remains unchanged: stabilize the existing voice-first architecture with narrow, production-critical fixes.
