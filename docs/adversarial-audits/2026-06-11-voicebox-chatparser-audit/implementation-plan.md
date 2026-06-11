# Implementation Plan

## Patch Set 1

- Target finding: CLI contract failures and Voicebox SSE terminal status drift.
- Root cause hypothesis:
  - CLI did not require `--input-directory` and invalid directory validation
    constructed but did not raise an error.
  - Voicebox SSE decoder returned the first status event, even when it was
    non-terminal.
- Goal:
  - Missing/invalid input exits through argparse with code `2`.
  - Progress bars are default-on for folder/file loops.
  - Generation waits until terminal SSE status before fetching audio.
- Files:
  - `chatparser.py`
  - `voicebox_client.py`
  - `tests/test_cli_contract.py`
  - `tests/test_voicebox_client.py`
- Dependencies / ordering:
  - Add failing tests before relying on the behavior in CLI smoke checks.
- Exact proof command:
  - `./.venv/bin/python -m pytest tests/test_voicebox_client.py tests/test_cli_contract.py`
- Rollback:
  - Revert if CLI no longer accepts documented valid input directory usage or
    queued generation happy path regresses.

## Patch Set 2

- Target finding: Text exports silently drop normal chat messages.
- Root cause hypothesis:
  - Parsed lines without audio attachments fell through to a diagnostic print
    rather than being appended to output.
- Goal:
  - Text export keeps original parsed messages and inserts transcriptions only
    where audio attachments are transcribed.
- Files:
  - `chatparser.py`
  - `tests/test_chatparser_voicebox.py`
- Exact proof command:
  - `./.venv/bin/python -m pytest tests/test_chatparser_voicebox.py`
- Rollback:
  - Revert if transcription insertion or Android/iOS message formatting
    regresses.

## Patch Set 3

- Target finding: Existing audio attachment conversion can fail silently.
- Root cause hypothesis:
  - Conversion happened in child worker processes whose exceptions were not
    returned to the parent; target directory and `.mp3` suffix handling were
    incomplete.
- Goal:
  - Attachment conversion creates `audio_out`, writes `<stem>.mp3`, and raises
    visible errors for missing source media.
- Files:
  - `chatparser.py`
  - `tests/test_chatparser_voicebox.py`
- Exact proof command:
  - `./.venv/bin/python -m pytest tests/test_chatparser_voicebox.py`
- Rollback:
  - Revert if large attachment conversion needs asynchronous worker behavior
    restored with explicit error propagation.

## Patch Set 4

- Target finding: Voicebox client accepts malformed profiles and non-audio
  generation downloads as successful.
- Root cause hypothesis:
  - Client treated unknown profile response shapes as empty lists and wrote any
    `2xx` `/audio` body to disk.
- Goal:
  - Unexpected `/profiles` shapes raise `VoiceboxError`.
  - Empty or non-audio `/audio` responses raise before output files are written.
- Files:
  - `voicebox_client.py`
  - `tests/test_voicebox_client.py`
- Exact proof command:
  - `./.venv/bin/python -m pytest tests/test_voicebox_client.py`
- Rollback:
  - Revert if current Voicebox returns an alternate documented profile wrapper
    or audio endpoint media type that needs explicit support.

## Patch Set 5

- Target finding: Docs contradicted current Voicebox multipart and readiness
  contracts.
- Root cause hypothesis:
  - Older docs preserved assumptions from README examples while the bundled
    backend route contract changed or was verified more precisely.
- Goal:
  - Docs say transcription multipart field is `file`.
  - Docs distinguish `/health` service readiness from `/profiles` profile
    availability.
- Files:
  - `docs/api.md`
  - `docs/runbook.md`
  - `docs/risk-register.md`
- Exact proof command:
  - `git diff --check`
- Rollback:
  - Revert or update if upstream Voicebox changes the route contract.
