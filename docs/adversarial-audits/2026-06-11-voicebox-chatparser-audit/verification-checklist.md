# Verification Checklist

- [x] Reproduction command captured.
- [x] Baseline captured.
- [x] Patch-set verification commands captured.
- [x] End-to-end Python verification run.
- [x] End-to-end Swift verification run.
- [x] CLI smoke verification run.
- [x] Residual risks documented.

## Final Verification

```text
./.venv/bin/python -m pytest
```

Result:

```text
37 passed, 2 skipped in 4.17s
```

```text
swift test
```

Result:

```text
4 Swift Testing tests passed
```

```text
git diff --check
```

Result: passed.

```text
./.venv/bin/python chatparser.py --input-directory <temp-export> --no-progress-bar
```

Result: exited `0`; generated `_chat-aud2txt.txt` preserved normal parsed chat
messages.

## Skipped / Not Available

- `python -m pytest -m voicebox_integration`: skipped by default because
  `VOICEBOX_SAMPLE_AUDIO` and `VOICEBOX_PROFILE_ID` are unset.
- Ruff: not installed in the repo `.venv`; no global installation attempted.
- Swift `VoiceboxAPI` URLProtocol contract tests: not implemented in this
  patch set.
