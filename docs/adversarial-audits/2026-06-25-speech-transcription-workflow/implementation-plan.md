# Implementation Plan

## Patch Set 1: Language hint propagation

- Target finding: configured transcription language was dropped.
- Root cause: `--voicebox-language` was only used for generation.
- Change:
  - Add `language` parameter to `VoiceboxClient.transcribe_audio()`.
  - Send `language` as multipart form data when present.
  - Pass `__VOICEBOX_LANGUAGE` from `chatparser.transcribe_audio_line()`.
  - Use configured language as display fallback when Voicebox omits detected
    language.
- Proof:
  `./.venv/bin/python -m pytest tests/test_voicebox_client.py::test_transcribe_audio_posts_optional_language_hint tests/test_chatparser_voicebox.py::test_transcribe_audio_line_uses_configured_language_when_voicebox_omits_it -q`
- Status: fixed.

## Patch Set 2: Video attachment routing

- Target finding: documented audio/video transcription skipped videos.
- Root cause: attachment classifier only treated audio names/extensions as
  transcribable.
- Change:
  - Recognize `VIDEO`, `VID-`, `.mp4`, `.mov`, `.m4v`, `.3gp`, and `.3gpp` as
    transcribable media.
  - Add parser tests proving iOS video attachments are routed to Voicebox.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_ios_video_attachment_is_detected_for_transcription tests/test_whatsapp_formats.py::test_process_ios_export_transcribes_video_attachment -q`
- Status: fixed.

## Patch Set 3: Contract documentation cleanup

- Target finding: CLI/docs overpromised prompt and response behavior.
- Root cause: legacy direct-Whisper wording remained after the Voicebox REST
  migration.
- Change:
  - README transcription example includes `--voicebox-language`.
  - CLI help states `--prompt-file` is legacy and not sent to Voicebox.
  - API, architecture, ADR, and runbook docs include optional `language`.
  - API docs describe current JSON response handling.
- Proof:
  `git diff --check`
- Status: fixed.

## Patch Set 4: Attachment parsing and batch isolation

- Target findings:
  - Android attachment filenames with spaces could resolve incorrectly.
  - Continuation-line attachment markers were skipped.
  - One transcription failure aborted the batch.
  - Attachment captions were discarded.
- Root cause:
  - Attachment matching was header-line-only and assumed no spaces in Android
    filenames.
  - Transcription errors were not isolated to the current media item.
  - Attachment metadata did not retain marker-adjacent caption text.
- Change:
  - Preserve full spaced WhatsApp audio/video filenames in Python and Swift.
  - Detect continuation-line attachment markers in Python text mode and Swift
    parsing.
  - Add safe text-mode transcription wrapper that records failure lines and
    continues.
  - Preserve captions in generated transcript/failure lines.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py tests/test_chatparser_voicebox.py -q`
- Status: fixed.

## Patch Set 5: macOS run guards

- Target findings:
  - Run could launch after Voicebox startup failed or the user cancelled during
    startup.
  - Invalid transcription model was accepted until Python argparse failed.
  - Cancel used SIGTERM and bypassed Python partial-output recovery.
- Root cause:
  - AppState did not track the startup task or require a startup success result
    before runner launch.
  - Swift model validation did not mirror Python choices.
  - Runner cancellation did not use the signal handled by Python.
- Change:
  - Add injectable runner/server/health-check seams.
  - Make Voicebox startup return readiness.
  - Stop before Python launch on startup failure or run-task cancellation.
  - Mirror valid transcription model choices in `RunConfiguration`.
  - Send SIGINT on runner cancel, with terminate fallback.
- Proof:
  `swift test`
- Status: fixed with bounded runtime residual for live partial-output signal
  verification.
