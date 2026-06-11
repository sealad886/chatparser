# Adversarial Audit Report

## Executive Summary

- Target: `chatparser` Python CLI, local Voicebox REST client, SwiftPM macOS
  shell, scripts, docs, and tests.
- Environment: macOS, repo `.venv`, branch `codex/voicebox-chat-workspace`.
- Done criteria: each in-scope confirmed finding is fixed with tests, proven
  not reproducible, or recorded as bounded residual risk.
- Highest-risk fixed findings:
  - Text export could silently produce an empty transformed file for normal
    parsed chat messages.
  - Audio export attachment conversion could fail in a child process without
    surfacing the error to the parent.
  - Queued Voicebox generation could see only the first non-terminal SSE event
    and time out despite a later terminal status.
- Current status: Python-side high-impact findings fixed and verified. Swift
  progress/timeout/API-contract coverage and large-file memory behavior remain
  bounded residual risks.

## System Map and Evidence Index

- Python CLI entrypoint: `chatparser.py`.
- Voicebox client boundary: `voicebox_client.py`.
- Swift app shell:
  - `Sources/ChatParserMac/Models/RunConfiguration.swift`
  - `Sources/ChatParserMac/Services/ChatParserRunner.swift`
  - `Sources/ChatParserMac/Services/VoiceboxAPI.swift`
  - `Sources/ChatParserMac/Services/VoiceboxServerController.swift`
- Vendored Voicebox contract evidence:
  - `external/voicebox/backend/routes/transcription.py`
  - `external/voicebox/backend/routes/generations.py`
  - `external/voicebox/backend/routes/audio.py`
- Tests:
  - `tests/test_cli_contract.py`
  - `tests/test_chatparser_voicebox.py`
  - `tests/test_voicebox_client.py`
  - `tests/test_whatsapp_formats.py`
  - `tests/ChatParserMacTests/ConversationAudioSuggestionsTests.swift`

See `evidence-index.md` for command evidence.

## Measurement Methodology

- Used repo-local `.venv` for all Python commands.
- Ran baseline and final Python and Swift test suites.
- Used current bundled Voicebox backend source as authoritative local contract
  evidence for endpoints.
- Used focused CLI smokes with temporary WhatsApp export folders to reproduce
  user-visible behavior.
- Subagents independently audited recon, correctness, contracts/docs, and
  runtime readiness.

## Baseline / Reproduction Matrix

| Area | Baseline evidence | Result |
|---|---|---|
| Python tests | `./.venv/bin/python -m pytest` | `24 passed, 2 skipped` |
| Swift tests | `swift test` | `4` tests passed |
| Missing input CLI | `./.venv/bin/python chatparser.py` | Unhandled `TypeError` |
| Plain text export | Temp export smoke | Empty `_chat-aud2txt.txt` |
| Voicebox SSE wait | Code/test inspection | First event returned even if non-terminal |
| Attachment conversion | Code/test inspection | Worker errors not propagated |

## Bottleneck / Risk Ranking

1. High: silent data omission in normal text export.
2. High: audio attachment conversion could fail without parent failure.
3. High: CLI missing/invalid input produced poor failure behavior.
4. Medium: queued Voicebox generation SSE handling could stall against real
   backend streams.
5. Medium: malformed `/profiles` and non-audio `/audio` responses were accepted
   too leniently.
6. Medium: docs drifted from current Voicebox route contract.
7. Medium residual: Swift import/generation progress and timeout behavior need
   deeper tests and UI work.
8. Medium residual: full-file chat and multipart media buffering can create
   memory pressure on very large exports.

## Detailed Findings

### Fixed: Text Export Dropped Normal Parsed Messages

- Severity: high.
- Affected paths: `chatparser.py::process_chat_file_by_type`.
- Evidence: temp export with two normal chat lines exited `0` but produced an
  empty `_chat-aud2txt.txt`.
- Root cause: parsed non-attachment messages in text mode fell through to a
  diagnostic print instead of being appended.
- Fix: added `format_parsed_whatsapp_line` and append normal parsed messages.
- Validation: `tests/test_chatparser_voicebox.py::test_text_export_preserves_non_attachment_messages`
  plus final CLI smoke.
- Status: fixed.

### Fixed: Audio Attachment Conversion Failure Was Hidden

- Severity: high.
- Affected paths: `chatparser.py::move_audio_file`,
  `chatparser.py::process_chat_file_by_type`.
- Evidence: worker process exceptions were not propagated; `audio_out` was not
  guaranteed; suffix logic could produce `clipmp3`.
- Root cause: conversion was queued to unmanaged worker processes and target
  path construction used string slicing.
- Fix: convert attachments synchronously, create `audio_out`, use
  `Path(filename).stem + ".mp3"`, and raise `FileNotFoundError` for missing
  source media.
- Validation:
  - `test_audio_export_converts_existing_attachment_with_visible_output`
  - `test_audio_export_missing_attachment_fails_visibly`
- Status: fixed.

### Fixed: CLI Input Contract Failed Poorly

- Severity: high.
- Affected paths: `chatparser.py` CLI parser.
- Evidence: running with no args raised `TypeError`; invalid directory
  validation constructed `AssertionError` without raising.
- Root cause: missing argparse requirement/guard and invalid validator.
- Fix: raise `argparse.ArgumentTypeError` for invalid directories and call
  `parser.error` when no input directory is supplied.
- Validation: `tests/test_cli_contract.py`.
- Status: fixed.

### Fixed: Voicebox Generation SSE Handling Returned Too Early

- Severity: medium.
- Affected paths: `voicebox_client.py::_decode_generation_status_event`.
- Evidence: bundled Voicebox status endpoint streams `generating` updates until
  terminal status.
- Root cause: SSE decoder returned first valid data event.
- Fix: consume events until terminal status or EOF, returning latest event only
  when the stream ends without a terminal status.
- Validation: `test_generate_speech_consumes_streaming_status_until_terminal_event`.
- Status: fixed.

### Fixed: Malformed Voicebox Responses Were Accepted

- Severity: medium.
- Affected paths: `voicebox_client.py::list_profiles`,
  `voicebox_client.py::generate_speech`.
- Evidence: profile dicts with missing/invalid `profiles` returned `[]`; any
  `2xx` `/audio` body was written to disk.
- Root cause: overly permissive response handling.
- Fix: raise on unexpected profile shapes, empty audio, and non-`audio/*`
  content types.
- Validation:
  - `test_list_profiles_rejects_unexpected_shapes`
  - `test_generate_speech_rejects_empty_audio_response`
  - `test_generate_speech_rejects_non_audio_response`
- Status: fixed.

### Fixed: Voicebox Docs Drift

- Severity: medium.
- Affected paths: `docs/api.md`, `docs/runbook.md`,
  `docs/risk-register.md`.
- Evidence: docs listed transcription field `audio` while code and bundled
  backend require `file`; docs disagreed on `/health` vs `/profiles`.
- Root cause: docs preserved older assumptions.
- Fix: document `file`, `/health` for service readiness, and `/profiles` for
  profile availability.
- Validation: `git diff --check`; endpoint evidence from bundled backend.
- Status: fixed.

## Prioritized Remediation Plan

Completed patch sets are documented in `implementation-plan.md`.

Recommended next patch sets:

1. Add Swift `URLProtocol` tests for `VoiceboxAPI` profile, generate, SSE,
   error, timeout, and audio body validation paths.
2. Add progress callbacks and ETA for Swift import/parsing loops.
3. Benchmark very large chat exports and large media uploads, then decide
   whether streaming file and multipart IO is required.
4. Clarify or implement the documented project artifact-root model versus the
   current CLI behavior of writing derived files beside the selected export.
5. Add a non-GUI build/readiness verification command separate from
   `script/build_and_run.sh --verify`.

## Validation Plan

Required before release:

- `./.venv/bin/python -m pytest`
- `swift test`
- `git diff --check`
- Real Voicebox integration tests with:
  - `VOICEBOX_SAMPLE_AUDIO`
  - `VOICEBOX_PROFILE_ID`
  - local Voicebox running
- Large-export smoke with nested folders and enough lines to verify progress
  and memory behavior.

## Residual Risks and Rollback Notes

- Accepted residual risk: Swift app import progress currently lacks ETA and
  detailed per-line/per-file progress.
- Accepted residual risk: Swift `VoiceboxAPI` lacks URLProtocol-backed contract
  tests for malformed/non-audio bodies and timeout classes.
- Accepted residual risk: Python CLI still reads whole chat files and stores
  output in memory before final write.
- Accepted residual risk: Python and Swift multipart assembly buffer whole media
  files in memory.
- Accepted residual risk: current CLI writes derived files into/near the input
  export folder; docs for future artifact-root storage should be reconciled with
  this behavior in a separate product decision.
- Rollback trigger: if local Voicebox returns valid audio with a blank or
  non-`audio/*` content type, adjust `_read_audio_response` to accept that
  specific proven contract rather than reverting all response validation.
