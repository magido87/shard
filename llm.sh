#!/usr/bin/env bash
set -euo pipefail

LOCALAI_DIR="$HOME/.localai"
VENV_PY="$LOCALAI_DIR/venv/bin/python"
CHAT_APP="$LOCALAI_DIR/chat.py"

if [ ! -x "$VENV_PY" ]; then
    echo "Missing Python venv: $VENV_PY"
    echo "Run: ~/.localai/setup.sh"
    exit 1
fi

if [ ! -f "$CHAT_APP" ]; then
    echo "Missing chat app: $CHAT_APP"
    exit 1
fi

exec "$VENV_PY" "$CHAT_APP" "$@"
