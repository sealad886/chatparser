#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICEBOX_DIR="$ROOT_DIR/external/voicebox"
BACKEND_DIR="$VOICEBOX_DIR/backend"
VENV_DIR="$BACKEND_DIR/venv"

cd "$ROOT_DIR"

if [[ ! -d "$VOICEBOX_DIR/.git" && ! -f "$VOICEBOX_DIR/.git" ]]; then
  git submodule update --init --recursive external/voicebox
fi

if [[ ! -d "$VOICEBOX_DIR" ]]; then
  echo "Voicebox submodule is missing at $VOICEBOX_DIR" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11+ is required to set up Voicebox." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
"$PIP" install --upgrade pip
"$PIP" install -r "$BACKEND_DIR/requirements.txt"

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  "$PIP" install -r "$BACKEND_DIR/requirements-mlx.txt"
  "$PIP" install --no-deps mlx-audio==0.4.1
  "$PIP" install --no-deps mlx-lm==0.31.1
fi

"$PIP" install --no-deps chatterbox-tts
"$PIP" install --no-deps hume-tada
"$PIP" install git+https://github.com/QwenLM/Qwen3-TTS.git

echo "Voicebox backend environment is ready at $VENV_DIR"
