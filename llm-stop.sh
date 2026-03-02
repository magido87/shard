#!/usr/bin/env bash
set -euo pipefail

PID_FILE="$HOME/.localai/llm.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No active LLM session found."
    exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "${pid:-}" ]; then
    rm -f "$PID_FILE"
    echo "Removed empty PID file."
    exit 0
fi

if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Removed stale PID file ($pid not running)."
    exit 0
fi

cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
if ! printf '%s' "$cmd" | grep -q ".localai/chat.py"; then
    echo "PID $pid is not .localai/chat.py. Refusing to kill unrelated process."
    echo "Command: ${cmd:-unknown}"
    exit 1
fi

kill "$pid" 2>/dev/null || true

for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
        break
    fi
    sleep 0.2
done

if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "Stopped LLM session (PID $pid)."
