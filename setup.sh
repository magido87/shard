#!/usr/bin/env bash
# localai — deploy to ~/.localai/ and ~/bin/
set -euo pipefail

LOCALAI_DIR="$HOME/.localai"
VENV_DIR="$LOCALAI_DIR/venv"
BIN_DIR="$HOME/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CY='\033[38;5;51m'; GR='\033[90m'; BD='\033[1m'
YL='\033[38;5;226m'; RD='\033[38;5;196m'; RS='\033[0m'
OK="${CY}✓${RS}"; FAIL="${RD}✗${RS}"

echo
echo -e "  ${BD}${CY}localai  ·  setup${RS}"
echo

# ── Check requirements ──────────────────────────────────────────
if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "  ${FAIL} localai requires Apple Silicon (M1 or later)."; exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo -e "  ${FAIL} python3 not found — run: xcode-select --install"; exit 1
fi

# ── Directories ─────────────────────────────────────────────────
mkdir -p "$LOCALAI_DIR" "$LOCALAI_DIR/logs" "$LOCALAI_DIR/plugins" "$BIN_DIR"
echo -e "  ${OK} directories ready"

# ── Venv ────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo -e "  ${OK} venv exists"
else
    echo -ne "  ${GR}creating venv …${RS}"
    python3 -m venv "$VENV_DIR"
    echo -e "\r  ${OK} venv created           "
fi

# ── Dependencies ────────────────────────────────────────────────
echo -ne "  ${GR}installing dependencies …${RS}"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q
echo -e "\r  ${OK} dependencies installed      "

# ── Copy Python files ────────────────────────────────────────────
if [ "$SCRIPT_DIR" != "$LOCALAI_DIR" ]; then
    cp "$SCRIPT_DIR"/*.py "$LOCALAI_DIR/"
    if [ -d "$SCRIPT_DIR/plugins" ]; then
        cp "$SCRIPT_DIR/plugins/"*.py "$LOCALAI_DIR/plugins/" 2>/dev/null || true
    fi
    echo -e "  ${OK} Python files + plugins copied to ${LOCALAI_DIR}"
fi

# ── ~/bin/llm ────────────────────────────────────────────────────
cat > "$BIN_DIR/llm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
LOCALAI_DIR="$HOME/.localai"
VENV_PY="$LOCALAI_DIR/venv/bin/python"
CHAT_APP="$LOCALAI_DIR/chat.py"
if [ ! -x "$VENV_PY" ]; then
    echo "Missing venv. Run: ~/.localai/setup.sh"; exit 1
fi
if [ ! -f "$CHAT_APP" ]; then
    echo "Missing chat.py. Run: ~/.localai/setup.sh"; exit 1
fi
exec "$VENV_PY" "$CHAT_APP" "$@"
EOF
chmod +x "$BIN_DIR/llm"
echo -e "  ${OK} ~/bin/llm"

# ── ~/bin/llm-stop ───────────────────────────────────────────────
cat > "$BIN_DIR/llm-stop" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PID_FILE="$HOME/.localai/llm.pid"
if [ ! -f "$PID_FILE" ]; then echo "No active session."; exit 0; fi
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "${pid:-}" ]; then rm -f "$PID_FILE"; exit 0; fi
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"; echo "Stale PID removed."; exit 0
fi
kill "$pid" 2>/dev/null || true
sleep 0.4
kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Stopped (PID $pid)."
EOF
chmod +x "$BIN_DIR/llm-stop"
echo -e "  ${OK} ~/bin/llm-stop"

# ── Done ─────────────────────────────────────────────────────────
echo
echo -e "  ${OK} ${BD}Done.${RS}  Make sure ${CY}~/bin${RS} is in your PATH, then type ${CY}llm${RS}."
echo -e "  ${GR}If llm is not found: ${CY}export PATH=\"\$HOME/bin:\$PATH\"${RS}"
echo
