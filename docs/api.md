# API

## Scope

This document defines ChatParser's internal service boundaries and the external
Voicebox REST dependency. The MVP is a local desktop app, so ChatParser does not
expose a public network API. If the macOS shell and processing core communicate
over a local helper process, the helper must bind to loopback only and use a
random per-session token.

## External Dependency: Voicebox REST

ChatParser integrates Voicebox through HTTP on loopback:

```text
Base URL: http://127.0.0.1:17493
```

Upstream README evidence:

- Speech generation: `POST /generate`
- Transcription: `POST /transcribe`
- Profile listing: `GET /profiles`
- Voicebox API docs when running: `http://127.0.0.1:17493/docs`
- Backend stack: FastAPI
- Database: SQLite
- Privacy model: local models, voice data, and captures

Source: <https://github.com/jamiepine/voicebox/blob/main/README.md>

### Health Check

ChatParser checks service readiness with a short-timeout `GET /health` call
when the bundled server starts. Profile-dependent generation workflows also
check `GET /profiles` because profile listing is side-effect free and proves
the profile store is reachable.

Expected handling:

| Outcome | ChatParser Behavior |
|---|---|
| 200 response | Mark Voicebox available and cache profile list briefly. |
| Connection refused | Show "Voicebox is not running" and keep job queued. |
| Timeout | Show retryable service timeout. |
| Non-2xx | Show endpoint error with status code, without dumping response body into logs. |

### `GET /profiles`

Purpose: list Voicebox voice profiles for generation.

Request:

```http
GET /profiles HTTP/1.1
Host: 127.0.0.1:17493
```

Response contract used by ChatParser:

```json
[
  {
    "id": "profile-id",
    "name": "Profile name",
    "language": "en"
  }
]
```

ChatParser must tolerate additional fields and missing optional labels. The
stable value persisted in `generated_audio.voicebox_profile_id` is the profile
id sent to `POST /generate`.

### `POST /transcribe`

Purpose: transcribe audio extracted from WhatsApp audio or video attachments.

Request:

```http
POST /transcribe HTTP/1.1
Host: 127.0.0.1:17493
Content-Type: multipart/form-data
```

Fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| file | File | Yes | Audio file, staged from project workspace. |
| model | Text | Yes | Voicebox model name, for example `turbo`. |

Example:

```bash
curl -X POST http://127.0.0.1:17493/transcribe \
  -F "file=@recording.wav" \
  -F "model=turbo"
```

ChatParser response handling:

- Accept JSON or text response shapes by using a typed adapter with schema
  guards.
- Extract transcript text into `transcript_version.transcript_text`.
- Store response metadata in `provenance_json`.
- Never overwrite previous transcript versions for the same attachment.
- Mark file-level failures and continue the batch unless user selected
  fail-fast.

### `POST /generate`

Purpose: generate speech audio from selected chat text, transcript text, or a
derived text artifact.

Request:

```http
POST /generate HTTP/1.1
Host: 127.0.0.1:17493
Content-Type: application/json
```

Body:

```json
{
  "text": "Hello world",
  "profile_id": "abc123",
  "language": "en"
}
```

ChatParser request constraints:

- `text` comes from a selected message, transcript, or local text derivation.
- `profile_id` must be selected from a recent `GET /profiles` result or entered
  manually by an advanced user.
- `language` defaults from project settings and can be overridden per request.

ChatParser response handling:

- Persist returned audio to `generated-audio/`.
- Record profile id, language, endpoint, request hash, output hash, and duration
  in `generated_audio.provenance_json`.
- Keep generated audio linked to the source text version so review decisions are
  reproducible.

## Internal App Service API

If ChatParser splits the UI from a local processing helper, these endpoints are
the local-only contract. They are not internet-facing.

### `POST /projects`

Create or open a project workspace.

Request:

```json
{
  "name": "Family WhatsApp Export",
  "artifact_root": "/Users/example/ChatParserArtifacts"
}
```

Response:

```json
{
  "project_id": "uuid",
  "workspace_path": "/Users/example/ChatParserArtifacts/projects/uuid"
}
```

### `POST /projects/{project_id}/imports`

Start import.

Request:

```json
{
  "source_path": "/Users/example/Downloads/WhatsApp Chat",
  "copy_mode": "copy"
}
```

Response:

```json
{
  "job_run_id": "uuid",
  "status": "queued"
}
```

### `POST /projects/{project_id}/transcriptions`

Start batch transcription.

Request:

```json
{
  "attachment_ids": ["uuid"],
  "model": "turbo",
  "fail_fast": false
}
```

Response:

```json
{
  "job_run_id": "uuid",
  "status": "queued"
}
```

### `POST /projects/{project_id}/generations`

Start generated-audio job.

Request:

```json
{
  "source_kind": "transcript_version",
  "source_id": "uuid",
  "profile_id": "abc123",
  "language": "en"
}
```

Response:

```json
{
  "job_run_id": "uuid",
  "status": "queued"
}
```

### `GET /jobs/{job_run_id}/events`

Replay job progress.

Response:

```json
{
  "job_run_id": "uuid",
  "status": "running",
  "completed_units": 42,
  "total_units": 100,
  "eta_seconds": 130,
  "events": [
    {
      "occurred_at": "2026-06-05T12:00:00Z",
      "level": "info",
      "phase": "transcribe",
      "message": "Transcribed attachment",
      "metadata": {
        "attachment_id": "uuid",
        "current_file": "PTT-20240501-WA0001.opus"
      }
    }
  ]
}
```

## Auth and Authorization

- MVP desktop app has a single local user and no remote account.
- Any local helper API must bind to `127.0.0.1` only.
- The shell must generate a random session token and pass it to helper requests.
- Token must not be written to project logs.
- File access is constrained by macOS picker-scoped paths and explicit artifact
  root selection.

## Error Model

| Code | Meaning | User Action |
|---|---|---|
| `VOICEBOX_UNAVAILABLE` | `127.0.0.1:17493` connection failed | Start Voicebox and retry. |
| `VOICEBOX_TIMEOUT` | Request timed out | Retry or lower batch concurrency. |
| `VOICEBOX_BAD_RESPONSE` | Response shape not understood | Open diagnostic details and preserve raw status metadata. |
| `SOURCE_FILE_MISSING` | Attachment referenced but absent | Continue import and mark attachment missing. |
| `UNSUPPORTED_MEDIA` | Media cannot be staged for Voicebox | Skip file or convert manually. |
| `ARTIFACT_WRITE_FAILED` | Local workspace write failed | Choose writable storage or free disk. |

## Versioning

- Persist Voicebox base URL and endpoint path with every transcript and
  generated-audio artifact.
- Persist model/profile/language values used per request.
- Add integration tests that validate documented endpoint contracts against a
  running Voicebox instance before changing the Voicebox client.
