# Phase 2 Self-Audit

Scope: audit the Phase 1 plan against the current code before research.

## Findings

### Blocking

1. Voice sessions are not caller-isolated.

- Code: `src/vaidya/api/routes/voice.py`
- `phone_hash = hashlib.sha256(b"twilio-call").hexdigest()[:16]` means all voice callers share one phone index.
- Impact: dropped-call recovery can resume the wrong caller's session, crossing PII and eligibility context.
- Required fix: derive `phone_hash` from a hashed Twilio caller parameter or call SID, never from a constant.

2. Twilio signature validation is incomplete.

- Code: `src/vaidya/api/routes/voice.py`
- The route checks only that `X-Twilio-Signature` exists.
- Impact: forged webhook requests can start streams or probe the service.
- Required fix: validate with Twilio `RequestValidator` when `TWILIO_AUTH_TOKEN` is configured.

3. Pipecat VAD wiring is incompatible with installed Pipecat.

- Code: `src/vaidya/telephony/pipeline.py`
- Local inspection of Pipecat 1.1 shows `FastAPIWebsocketParams` has no `vad_analyzer` argument.
- Impact: real voice calls can crash at pipeline construction when `VAD_AVAILABLE` is true.
- Required fix: use Sarvam STT `vad_signals=True` rather than passing local VAD into transport params.

4. Conversation route loses channel semantics.

- Code: `src/vaidya/api/routes/conversation.py`
- The route validates `request.channel` but calls `manager.start_conversation(..., language=validated_lang)` without `channel=request.channel`.
- Impact: API clients asking for voice onboarding receive text onboarding.
- Required fix: pass the channel through and cover it with tests.

### High Priority

5. Raw caller PII is passed and logged.

- Code: `incoming_call()` logs `caller` and passes it as a Twilio Stream parameter.
- Impact: phone numbers can appear in app logs and WebSocket custom parameters.
- Required fix: log and pass only a hash.

6. Welcome silence reprompt under-represents supported languages.

- Code: `src/vaidya/i18n/strings/orchestrator.json`
- `silence_welcome_reprompt` says only "Hindi, Tamil, or English".
- Impact: users of the other 8 voice languages are nudged toward the wrong set.
- Required fix: concise copy mentioning the full supported set or "say your language name" with examples.

7. Idle watcher lacks explicit cleanup.

- Code: `src/vaidya/telephony/agent_processor.py`
- The idle loop exits when `_wake` is set, but processor cleanup does not explicitly wake/cancel.
- Impact: low but avoidable dangling task risk at call teardown.
- Required fix: implement processor `cleanup()`.

### Medium Priority

8. Intake prompt wording is internally contradictory.

- Code: `src/vaidya/prompts/templates/intake_system.txt`
- It says "exactly ONE short word" but examples include "Theek hai".
- Impact: inconsistent LLM acknowledgements and potentially verbose TTS.
- Required fix: reword to "one short acknowledgement phrase".

9. `VaidyaAgentProcessor` duplicates TTS speaker mapping.

- Code: `src/vaidya/telephony/agent_processor.py`
- The map mirrors `vaidya.voice.language.TTS_SPEAKERS`.
- Impact: drift risk as languages/speakers change.
- Required fix: import the central mapping directly if it does not create a cycle.

10. Existing untracked `.env.production` must stay uncommitted.

- Code: repository root.
- Impact: potential secret leak.
- Required fix: avoid staging it and verify final status before commits.

## Audit Result

The Phase 1 plan is implementable and aligned with the codebase, but it must be revised to explicitly account for Pipecat 1.1's actual constructor signatures and for Twilio session identity moving before conversation creation.
