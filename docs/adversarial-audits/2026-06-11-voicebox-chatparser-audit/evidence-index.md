# Evidence Index

## Commands

Baseline before remediation:

- `./.venv/bin/python -m pytest`
  - Result: `24 passed, 2 skipped in 8.50s`.
- `swift test`
  - Result: `4` Swift Testing tests passed.
- `./.venv/bin/python chatparser.py`
  - Result: exited `1` with `TypeError: 'NoneType' object is not iterable`.
- Temp CLI text-export smoke before patch:
  - Result: exited `0`, printed `Don't know what to do with this line 0`, and
    created an empty `_chat-aud2txt.txt`.

Patch verification:

- `./.venv/bin/python -m pytest tests/test_chatparser_voicebox.py tests/test_voicebox_client.py tests/test_cli_contract.py`
  - Result: `30 passed in 4.23s`.
- `git diff --check`
  - Result: passed.
- `./.venv/bin/python -m pytest`
  - Result: `37 passed, 2 skipped in 4.17s`.
- `swift test`
  - Result: `4` Swift Testing tests passed.
- Temp CLI text-export smoke after patch:
  - Result: exited `0`; `_chat-aud2txt.txt` preserved normal parsed chat
    messages.

Tooling not available:

- `./.venv/bin/python -m ruff --version`
  - Result: `No module named ruff`; no global install attempted.

## Files and Symbols

- `chatparser.py`
  - `process_chat_file_by_type`: main transform loop.
  - `format_parsed_whatsapp_line`: preserves normal parsed messages in text
    export output.
  - `move_audio_file`: converts existing audio attachments into
    `audio_out/<stem>.mp3` and fails visibly when source media is missing.
  - CLI parser: now rejects missing/invalid `--input-directory`, defaults
    progress bars on, and exposes `--no-progress-bar`.
- `voicebox_client.py`
  - `VoiceboxClient.list_profiles`: rejects malformed profile response shapes.
  - `VoiceboxClient.generate_speech`: validates non-empty `audio/*` download
    before writing output.
  - `_decode_generation_status_event`: consumes SSE until terminal status.
- `tests/test_cli_contract.py`
  - Regression coverage for missing and invalid input directory behavior.
- `tests/test_chatparser_voicebox.py`
  - Regression coverage for normal text preservation, existing attachment
    conversion, and missing attachment failure.
- `tests/test_voicebox_client.py`
  - Regression coverage for terminal SSE status, malformed `/profiles`, and
    empty/non-audio `/audio` responses.
- `docs/api.md`, `docs/runbook.md`, `docs/risk-register.md`
  - Contract wording aligned to current Voicebox `file` multipart field and
    `/health` readiness plus `/profiles` availability checks.

## External Contract Evidence

- `external/voicebox/backend/routes/transcription.py`
  - `POST /transcribe` accepts multipart field `file`.
- `external/voicebox/backend/routes/generations.py`
  - `GET /generate/{generation_id}/status` streams SSE updates until terminal
    status.
- `external/voicebox/backend/routes/audio.py`
  - `/audio/{generation_id}` serves `FileResponse` with an audio media type
    derived from the generated file extension.
- `external/voicebox/backend/models.py`
  - `GenerationRequest` requires `profile_id`, `text`, and validates language
    and model fields.

## Artifacts

- Audit ledger:
  - `docs/adversarial-audits/2026-06-11-voicebox-chatparser-audit/audit-report.md`
  - `docs/adversarial-audits/2026-06-11-voicebox-chatparser-audit/evidence-index.md`
  - `docs/adversarial-audits/2026-06-11-voicebox-chatparser-audit/implementation-plan.md`
  - `docs/adversarial-audits/2026-06-11-voicebox-chatparser-audit/verification-checklist.md`

## Notes

- Real Voicebox integration tests remained skipped because
  `VOICEBOX_SAMPLE_AUDIO` and `VOICEBOX_PROFILE_ID` were not configured in this
  environment.
- Swift `VoiceboxAPI` contract tests were not added in this patch set; this is
  tracked as residual risk.
