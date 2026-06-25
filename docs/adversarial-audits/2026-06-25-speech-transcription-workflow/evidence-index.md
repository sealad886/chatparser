# Evidence Index

## Local Contract Evidence

- `external/voicebox/backend/routes/transcription.py`
  - `POST /transcribe` accepts multipart `file`, optional `language`, and
    optional `model`.
  - The route validates model size, downloads missing Whisper models with HTTP
    `202`, and returns `TranscriptionResponse`.
- `external/voicebox/backend/models.py`
  - `TranscriptionResponse` contains `text` and `duration`.
- `external/voicebox/backend/backends/mlx_backend.py`
  - MLX STT passes a provided language hint into Whisper generation options.
- `external/voicebox/backend/backends/pytorch_backend.py`
  - PyTorch STT maps a provided language hint into Whisper decoder prompt ids.

## External Documentation Checked

- Voicebox official docs describe a Python FastAPI backend with routes
  delegating through services/backends and automatic Whisper model downloads.
- Voicebox backend README lists `/transcribe` as the Whisper audio-to-text API
  domain and `/generate` status as SSE.
- Voicebox GitHub README confirms `/transcribe` is backed by Whisper and may
  drift from the bundled submodule example syntax.

## Reproductions

- Red test before patch:
  `VoiceboxClient.transcribe_audio(..., language="fr")` raised `TypeError`.
- Red test before patch:
  `chatparser.transcribe_audio_line()` called the fake client with no language
  and displayed `(unk)` when Voicebox omitted language.
- Red test before patch:
  iOS `VIDEO-...mp4` attachments were not marked for transcription.
- Subagent probe:
  `WhatsApp Audio 2024-12-31 at 23.05.00.opus (file attached)` was truncated
  by the old Android regex.
- Subagent probe:
  a continuation-line `<attached: clip.ogg>` marker produced no Voicebox call.
- Subagent probe:
  a fake first-file transcription error stopped the second attachment and left
  no processed output.

## Proof Commands

- Focused red/green:
  `./.venv/bin/python -m pytest tests/test_voicebox_client.py::test_transcribe_audio_posts_optional_language_hint tests/test_chatparser_voicebox.py::test_transcribe_audio_line_uses_configured_language_when_voicebox_omits_it tests/test_whatsapp_formats.py::test_process_android_export_transcribes_audio_attachment -q`
- Video regression:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py::test_ios_video_attachment_is_detected_for_transcription tests/test_whatsapp_formats.py::test_process_ios_export_transcribes_video_attachment -q`
- Attachment and failure regression:
  `./.venv/bin/python -m pytest tests/test_whatsapp_formats.py tests/test_chatparser_voicebox.py -q`
- Focused workflow suite:
  `./.venv/bin/python -m pytest tests/test_voicebox_client.py tests/test_chatparser_voicebox.py tests/test_whatsapp_formats.py tests/test_cli_contract.py -q`
- Full Python suite:
  `./.venv/bin/python -m pytest -q`
- Swift suite:
  `swift test`
- Whitespace check:
  `git diff --check`

## Worktree State

- Pre-existing unrelated untracked path preserved:
  `unsloth/`.
- No global Python packages were installed.
