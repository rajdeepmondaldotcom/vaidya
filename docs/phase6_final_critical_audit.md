# Phase 6 Final Critical Audit

Scope: final audit of the Phase 5 plan against the current code before implementation.

## Must Fix Before Shipping

1. `voice_stream()` starts sessions before Twilio handshake data is available.

- Required change: parse handshake first, derive `phone_hash`, then start conversation.
- Integration point: `run_voice_pipeline()` must accept already parsed `transport_type` and `call_data`.

2. `run_voice_pipeline()` currently owns Twilio handshake parsing unconditionally.

- Required change: make parsing optional so tests and future callers can still use the old entry shape.
- Integration point: serializer creation needs `stream_sid` and `call_sid` from the parsed data either way.

3. `incoming_call()` needs three safety changes in one route.

- Required changes: real Twilio signature validation, hash caller before logging/streaming, XML escaping for attribute values.
- Integration point: `python-multipart` is already in dependencies for form parsing; `twilio` is in the telephony extra.

4. Pipecat transport params will fail with local VAD.

- Required change: remove unsupported `vad_analyzer` param and configure Sarvam STT VAD signals instead.
- Integration point: existing `VaidyaAgentProcessor` already reacts to `UserStartedSpeakingFrame`.

5. HTTP conversation start route must pass channel.

- Required change: `manager.start_conversation(..., channel=request.channel)`.

6. I18n and prompt copy must be updated safely.

- Required changes: full language set in welcome silence reprompt, non-contradictory acknowledgement instruction.
- Integration point: existing tests assert old Hindi/Tamil/English substring; update to assert broader supported language behavior.

7. Idle task lifecycle must be explicit.

- Required change: implement async `cleanup()` in `VaidyaAgentProcessor` and wake/cancel the idle watcher.

## Tests To Add Or Update

- Conversation route test for channel pass-through.
- Voice route tests for:
  - valid Twilio signature accepts and emits hashed custom parameter,
  - invalid Twilio signature rejects,
  - missing `voice_websocket_url` returns configured apology,
  - status callback attributes appear when configured,
  - no raw caller value appears in TwiML.
- Pipeline construction helper test for current Pipecat param compatibility.
- Agent processor test for central language normalization, including Pipecat `Language.OR_IN` -> `od-IN`.
- Existing silence tests updated for full language prompt.

## Final Implementation Sequence

1. Fix route-level Twilio security, identity, and channel pass-through.
2. Fix Pipecat/Sarvam pipeline compatibility and handshake ownership.
3. Fix voice processor cleanup and language mapping drift.
4. Tighten i18n/prompt copy.
5. Add focused tests.
6. Run targeted checks, then broad unit/integration checks.
7. Commit implementation in coherent units.
8. Produce post-implementation audit and fix anything it surfaces.

## Final Risk Notes

- The code can be made production-safer, but real "perfect" voice UX needs call recordings, latency traces, and field testing across devices, accents, and network conditions. The implementation will add the hooks and deterministic behavior needed for that process.
- Twilio signature validation can fail behind a proxy if the app sees a different URL than Twilio signed. Deployment must preserve forwarded scheme/host or configure the ASGI server/proxy accordingly.
- Sarvam STT/TTS quality depends on matching 8 kHz telephony audio configuration end-to-end.
