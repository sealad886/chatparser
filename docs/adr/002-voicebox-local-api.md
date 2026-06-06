# ADR-002: Voicebox Local API

Date: 2026-06-05
Status: Proposed

## Context

ChatParser must transcribe WhatsApp audio/video attachments and generate speech
from selected text. Voicebox already provides local speech-to-text and
text-to-speech capabilities. The upstream README documents a REST API at
`http://127.0.0.1:17493`, including `POST /transcribe`,
`POST /generate`, and `GET /profiles`. It also states that models, voice data,
and captures stay local, and identifies FastAPI and SQLite as backend storage
technologies.

Directly importing Whisper or Voicebox internals would couple ChatParser to
Voicebox implementation details and model runtime dependencies.

## Decision

Integrate Voicebox exclusively through its local REST API. ChatParser will:

- call `GET /profiles` for service availability and profile selection,
- call `POST /transcribe` with multipart `audio` and `model` fields,
- call `POST /generate` with JSON `text`, `profile_id`, and `language`,
- persist endpoint, model/profile/language, request provenance, and output
  checksums with each derived artifact,
- avoid direct Whisper imports and avoid reading/writing Voicebox SQLite.

## Alternatives Considered

- Direct Whisper imports in ChatParser - Rejected because it duplicates
  Voicebox responsibility, adds model/runtime complexity, and violates the
  requested integration boundary.
- Direct Voicebox Python module imports - Rejected because internal APIs may
  change and would bypass the documented REST contract.
- Cloud transcription provider - Rejected because the product is local-first and
  WhatsApp exports are highly sensitive.
- Voicebox SQLite integration - Rejected because Voicebox owns its storage and
  ChatParser only needs profile ids, transcript results, and generated audio.

## Consequences

- ChatParser remains smaller and avoids bundling STT/TTS model runtimes.
- Voicebox can evolve internally as long as the REST contract remains stable.
- Users must run Voicebox locally before transcription or generation.
- API drift must be caught with integration tests against the running local
  Voicebox service.
- A malicious or stale local service on the same port remains a residual risk
  and is handled through response validation and user-visible service state.

## References

- Voicebox README: <https://github.com/jamiepine/voicebox/blob/main/README.md>
- [API](../api.md)
- [Security](../security.md)
