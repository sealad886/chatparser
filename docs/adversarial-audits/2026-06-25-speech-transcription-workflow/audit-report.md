# Speech Transcription Workflow Adversarial Audit

Date: 2026-06-25

## Scope

- Target: ChatParser speech transcription path for WhatsApp media attachments.
- Runtime: repo-local `.venv`, bundled `external/voicebox` backend contract, and
  SwiftPM macOS shell command construction.
- Done criteria: confirmed current workflow defects are fixed with regression
  tests, or explicitly bounded as residual risk.

## System Map

- Python CLI entrypoint: `chatparser.py`.
- Voicebox REST client: `voicebox_client.py`.
- macOS app runner: `Sources/ChatParserMac/Models/RunConfiguration.swift` and
  `Sources/ChatParserMac/Services/ChatParserRunner.swift`.
- Bundled Voicebox transcription route:
  `external/voicebox/backend/routes/transcription.py`.
- Regression tests:
  `tests/test_voicebox_client.py`, `tests/test_chatparser_voicebox.py`, and
  `tests/test_whatsapp_formats.py`.
- Swift parser and runner tests:
  `tests/ChatParserMacTests/WhatsAppExportParserTests.swift`,
  `tests/ChatParserMacTests/RunConfigurationTests.swift`, and
  `tests/ChatParserMacTests/AppStateRunTests.swift`.

## Findings

### Fixed: configured transcription language was dropped

- Severity: high.
- Evidence: `RunConfiguration` and CLI expose `--voicebox-language`, and the
  bundled Voicebox `/transcribe` route accepts a multipart `language` field, but
  `VoiceboxClient.transcribe_audio()` only sent `model` and `file`.
- Root cause: ChatParser treated language as generation-only even though
  Voicebox STT supports it as a Whisper language hint.
- Fix: `VoiceboxClient.transcribe_audio()` now accepts optional `language`;
  `chatparser.transcribe_audio_line()` passes `__VOICEBOX_LANGUAGE` and uses it
  as the displayed fallback when Voicebox returns no detected language.
- Proof:
  `./.venv/bin/python -m pytest tests/test_voicebox_client.py::test_transcribe_audio_posts_optional_language_hint tests/test_chatparser_voicebox.py::test_transcribe_audio_line_uses_configured_language_when_voicebox_omits_it tests/test_whatsapp_formats.py::test_process_android_export_transcribes_audio_attachment -q`
- Status: fixed.

### Fixed: video attachments were not transcribed despite the workflow contract

- Severity: medium.
- Evidence: project docs describe audio/video transcription, while
  `_is_audio_attachment()` excluded `VIDEO-*`, `VID-*`, `.mp4`, `.mov`, `.m4v`,
  and mobile video containers, so those attachments were preserved as plain
  lines instead of sent to Voicebox.
- Root cause: attachment classification only recognized voice-note style audio
  names and audio extensions.
- Fix: video attachment names and common video suffixes are now treated as
  transcribable media in the Python text-transcription workflow.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_ios_video_attachment_is_detected_for_transcription tests/test_whatsapp_formats.py::test_process_ios_export_transcribes_video_attachment -q`
- Status: fixed.

### Fixed: prompt-file option overpromised ASR behavior

- Severity: medium.
- Evidence: CLI help described `--prompt-file` as transcription context, but
  the bundled Voicebox `/transcribe` route accepts `file`, `model`, and
  `language`, not a prompt field.
- Root cause: legacy Whisper-era CLI help survived the Voicebox REST migration.
- Fix: CLI help and README now state that `--voicebox-language` is the supported
  ASR hint and `--prompt-file` is not sent to Voicebox.
- Proof: `git diff --check` plus focused transcription tests.
- Status: fixed.

### Fixed: Android attachment filenames with spaces could resolve incorrectly

- Severity: high.
- Evidence: Android attachment parsing used `\S+\.[A-Za-z0-9]+`, so a filename
  like `WhatsApp Audio 2024-12-31 at 23.05.00.opus (file attached)` could be
  truncated to `23.05.00.opus` and transcribe the wrong file if that basename
  existed.
- Root cause: the parser assumed attachment filenames never contain spaces.
- Fix: Python and Swift parsers now recognize common spaced WhatsApp audio/video
  filename shapes and preserve the full basename.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_android_audio_attachment_with_spaces_preserves_full_filename tests/test_whatsapp_formats.py::test_process_android_export_transcribes_spaced_audio_filename -q`
  and `swift test`.
- Status: fixed.

### Fixed: continuation-line attachment markers were skipped

- Severity: high.
- Evidence: attachment detection only ran on successfully parsed header lines;
  a marker on the next line after a message header was appended as raw text and
  never sent to Voicebox.
- Root cause: continuation-line handling appended text before checking for
  attachment markers.
- Fix: Python text mode checks continuation lines for media markers using the
  previous message timestamp/speaker, and Swift parser attaches continuation
  markers to the current message.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_process_export_transcribes_continuation_attachment_marker -q`
  and `swift test`.
- Status: fixed.

### Fixed: one transcription failure aborted the batch

- Severity: high.
- Evidence: `transcribe_audio_line()` ran inside the main loop, and
  `cleanup_end()` only ran after the whole file; one `VoiceboxError`,
  missing-file error, or malformed response could stop later attachments and
  leave no processed output.
- Root cause: file-level transcription errors were not isolated from the batch.
- Fix: text transcription now records a `[Transcription failed]` line for the
  failed media item and continues with the remaining attachments.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_process_export_continues_after_transcription_failure -q`.
- Status: fixed.

### Fixed: attachment captions were discarded

- Severity: medium.
- Evidence: message text around an attachment marker was dropped once a
  transcript line was generated.
- Root cause: the attachment parser returned only filename/type, and transcript
  formatting ignored marker-adjacent text.
- Fix: parsed attachments now carry caption text with the marker removed; the
  generated transcript/failure line preserves that caption.
- Proof:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_process_export_preserves_attachment_caption_with_transcript -q`.
- Status: fixed.

### Fixed: macOS run could proceed after startup failure or startup cancel

- Severity: high.
- Evidence: `AppState.run()` awaited Voicebox startup but always launched the
  Python runner afterward; `cancel()` only terminated an already-started Python
  process, so cancel during Voicebox readiness wait could still launch
  ChatParser.
- Root cause: Voicebox startup returned no success/failure value and the run
  task was not tracked for cancellation.
- Fix: startup now returns a Boolean readiness result, `run()` stops before
  Python launch if Voicebox is unavailable or the run task was cancelled, and
  AppState has injectable runner/server/health-check seams for regression
  tests.
- Proof: `swift test` with `AppStateRunTests`.
- Status: fixed.

### Fixed: invalid transcription model launched Python before failing

- Severity: medium.
- Evidence: the macOS settings field allowed arbitrary model text while Python
  argparse only accepts `base`, `small`, `medium`, `large`, `turbo`, and the
  matching `whisper-*` aliases.
- Root cause: Swift UI validation did not mirror the Python CLI contract.
- Fix: `RunConfiguration.isTranscriptionModelValid` mirrors Python choices,
  `AppState.canRun` rejects invalid transcription models, and the UI shows a
  concise hint.
- Proof: `swift test` with `RunConfigurationTests`.
- Status: fixed.

### Fixed: macOS cancel bypassed Python partial-output recovery

- Severity: medium.
- Evidence: Python writes `_transcription_interrupted.txt` on
  `KeyboardInterrupt`, but the Swift runner used `Process.terminate()`
  (`SIGTERM`).
- Root cause: app cancellation did not map to the signal the Python CLI handles.
- Fix: `ChatParserRunner.cancel()` sends `SIGINT` first, falling back to
  terminate only if the signal call fails.
- Proof: `swift test` build/run coverage; no live long-running transcription
  fixture was available in this environment.
- Status: fixed with bounded runtime residual.

### Fixed: API docs overstated response-shape support

- Severity: low.
- Evidence: docs said ChatParser accepts JSON or text transcription responses,
  while the current client decodes JSON and the bundled route returns JSON.
- Root cause: future-facing API text did not match current client behavior.
- Fix: docs now describe the JSON response shape from the bundled backend.
- Proof: `git diff --check`.
- Status: fixed.

## Residual Risks

- Real video transcription still depends on the local Voicebox/librosa/ffmpeg
  decoder stack accepting the chosen video container. This patch proves routing
  and API contract behavior with tests, not media decoding quality.
- The SIGINT cancellation path is built and covered at the runner API level, but
  a live slow-transcription fixture was not available to prove the Python
  partial-output file is written after macOS Cancel.
- The official Voicebox `main` README currently differs from the bundled route
  source on the exact `/transcribe` multipart example. ChatParser uses the
  bundled submodule route as the compatibility oracle because
  `script/setup_voicebox.sh` prepares that backend.
- Integration tests that hit a running Voicebox service remain skipped unless
  `VOICEBOX_BASE_URL` and `VOICEBOX_SAMPLE_AUDIO` are configured.
- Future project-store docs still describe transcript versions/provenance that
  the current CLI does not implement; current CLI behavior remains a flat
  `-aud2txt` output file.

## Verification Summary

- `./.venv/bin/python -m pytest -q`: `58` passed, `2` skipped.
- `swift test`: `26` Swift Testing tests passed.
- `git diff --check`: passed.
- `./.venv/bin/python -m ruff ...`: not run; `.venv` has no `ruff` installed
  and no global install was attempted.
