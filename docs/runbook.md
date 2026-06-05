# Runbook

## Operating Model

Chatparser runs as a local macOS app. Voicebox runs as a separate local app or
backend service and must be reachable at `http://127.0.0.1:17493` for
transcription and generation. No cloud service is required for the MVP.

## Preflight

Before import or batch processing:

1. Confirm project artifact root is writable.
2. Estimate import size and free disk space.
3. Confirm source path exists and is readable.
4. Confirm Voicebox availability with `GET http://127.0.0.1:17493/profiles`
   when transcription or generation is requested.
5. Confirm selected transcription model is configured, for example
   `whisper-turbo`.
6. Confirm generated-audio requests have a profile id from `GET /profiles`.

Manual Voicebox checks:

```bash
curl http://127.0.0.1:17493/profiles

curl -X POST http://127.0.0.1:17493/transcribe \
  -F "audio=@recording.wav" \
  -F "model=whisper-turbo"

curl -X POST http://127.0.0.1:17493/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "profile_id": "abc123", "language": "en"}'
```

## Import Procedure

1. Create or open project.
2. Select WhatsApp export ZIP or folder.
3. Start import job.
4. Watch progress: phase, completed files/lines, total files/lines, ETA, current
   path basename.
5. Review import summary:
   - message count,
   - participant count,
   - available attachments,
   - missing attachments,
   - unsupported files,
   - parse warnings.
6. Resolve warnings or continue to transcription.

Expected artifacts:

- `import/source-manifest.json`
- `chatparser.sqlite`
- `media/original/` or source references depending on copy mode
- `logs/<job-id>.jsonl`

## Transcription Procedure

1. Start Voicebox.
2. Confirm `GET /profiles` responds.
3. Select attachments and model.
4. Start transcription job.
5. Progress must show completed attachments, total attachments, ETA, current
   file basename, retry count when applicable.
6. Review transcript versions and mark each as reviewed, edited, rejected, or
   left machine-generated.

Failure handling:

| Failure | Action |
|---|---|
| Voicebox connection refused | Start Voicebox and retry queued job. |
| Voicebox timeout | Retry file, lower concurrency, or use smaller model. |
| Unsupported media | Convert to accepted audio format or skip. |
| Disk write failure | Free disk, choose different artifact root, retry. |
| Malformed response | Save status metadata, skip file, open diagnostic issue. |

## Audio Generation Procedure

1. Start Voicebox.
2. Refresh profile list with `GET /profiles`.
3. Select text source, profile id, and language.
4. Start generation job.
5. Store output under `generated-audio/`.
6. Verify provenance label before export.

Generated audio must never replace original WhatsApp media. It is always a
derived artifact with source text and Voicebox profile provenance.

## Export Procedure

1. Select export format and destination.
2. Preview included conversations, attachments, transcripts, generated audio,
   and redactions.
3. Create export bundle with manifest and checksums.
4. Open export folder only after job completes.

Export bundle should include:

- content file such as `index.html`, `messages.json`, or `messages.csv`,
- selected media,
- selected transcripts,
- selected generated audio,
- `provenance.json`,
- `checksums.sha256`.

## Backup and Restore

Backup:

1. Quit active jobs or wait for completion.
2. Copy the whole project directory under `projects/<project-id>/`.
3. Optionally export a portable bundle for selected records.

Restore:

1. Place project directory under artifact root.
2. Open project from app.
3. Run integrity check to validate SQLite, artifact paths, and checksums.
4. Reconnect missing source paths only if user wants source media references.

## Observability

Job logs are append-only JSONL. Each event should include:

- job id,
- project id,
- phase,
- level,
- completed units,
- total units,
- ETA seconds where known,
- current file basename,
- endpoint for Voicebox calls,
- model/profile metadata,
- status or error code.

Logs must not include full chat message body, full transcript text, generated
text, or full source path by default.

## Incident Response

| Incident | Immediate Action | Follow-Up |
|---|---|---|
| Source export accidentally modified | Stop job, compare source checksums, restore from original export, fix import code. |
| Sensitive data found in logs | Stop diagnostic sharing, delete affected logs, patch logger, add regression test. |
| Generated audio mislabeled | Remove export, correct provenance rendering, regenerate bundle. |
| Voicebox remote URL used accidentally | Stop job, record endpoint, notify user, update DPIA/security review. |
| Project database corruption | Stop app, preserve copy, restore from backup or rebuild from source manifest. |

## Release Readiness Checks

- Import smoke test on a WhatsApp text export with attachments.
- Transcription integration test against running Voicebox `/transcribe`.
- Generation integration test against running Voicebox `/generate`.
- Voicebox unavailable test.
- Source checksum unchanged after import.
- Log scan confirms no raw message body or transcript text.
- Export manifest includes checksums and provenance.
- Folder/file loops show progress and ETA.
