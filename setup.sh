#!/usr/bin/env bash
set -euo pipefail

LOCALAI_DIR="$HOME/.localai"
VENV_DIR="$LOCALAI_DIR/venv"
BIN_DIR="$HOME/bin"
ZSHRC="$HOME/.zshrc"
MODEL="${MODEL:-mlx-community/Dolphin3.0-Llama3.1-8B-4bit}"
MARKER_START="# >>> localai-commands >>>"
MARKER_END="# <<< localai-commands <<<"

echo "=== Local AI setup (MLX + Dolphin 3.0 8B) ==="
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3 first."
    exit 1
fi

# 1) Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "[1/5] Python venv exists: $VENV_DIR"
else
    echo "[1/5] Creating Python venv..."
    mkdir -p "$LOCALAI_DIR"
    python3 -m venv "$VENV_DIR"
fi

# 2) Install dependencies
echo "[2/5] Installing dependencies (mlx-lm, psutil)..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install --upgrade mlx-lm psutil

# 3) Preload model into HuggingFace cache
echo "[3/5] Preloading model cache: $MODEL"
echo "      First run may download several GB."
"$VENV_DIR/bin/python" - <<PY
from mlx_lm import load
model_id = "${MODEL}"
print(f"      Loading {model_id} ...")
load(model_id)
print("      Model cached.")
PY

# 4) Write command wrappers in ~/bin
echo "[4/5] Installing command wrappers in $BIN_DIR ..."
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/llm" <<'EOF'
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
EOF

cat > "$BIN_DIR/llm-stop" <<'EOF'
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
EOF

chmod +x "$BIN_DIR/llm" "$BIN_DIR/llm-stop"

# 5) Idempotent zshrc block injection
echo "[5/5] Ensuring llm and llm-stop functions in $ZSHRC ..."
touch "$ZSHRC"
tmp="$(mktemp)"
awk -v s="$MARKER_START" -v e="$MARKER_END" '
    $0 == s { skip=1; next }
    $0 == e { skip=0; next }
    !skip { print }
' "$ZSHRC" > "$tmp"
mv "$tmp" "$ZSHRC"

{
    echo
    echo "$MARKER_START"
    echo 'llm() {'
    echo '  "$HOME/bin/llm" "$@"'
    echo '}'
    echo
    echo 'llm-stop() {'
    echo '  "$HOME/bin/llm-stop" "$@"'
    echo '}'
    echo
    echo 'alias llmstop="llm-stop"'
    echo "$MARKER_END"
} >> "$ZSHRC"

echo
echo "Setup complete."
echo "Open a new terminal tab, then run: llm"
echo "If needed, stop session with: llm-stop"
