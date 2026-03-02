#!/usr/bin/env bash
set -euo pipefail

LOCALAI_DIR="$HOME/.localai"
VENV_DIR="$LOCALAI_DIR/venv"
BIN_DIR="$HOME/bin"
ZSHRC="$HOME/.zshrc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER_START="# >>> localai-commands >>>"
MARKER_END="# <<< localai-commands <<<"

VOICE_OPT=0
for arg in "$@"; do
    case "$arg" in --voice) VOICE_OPT=1 ;; esac
done

CY='\033[38;5;51m'; GR='\033[90m'; BD='\033[1m'
YL='\033[38;5;226m'; RD='\033[38;5;196m'; RS='\033[0m'
OK="${CY}✓${RS}"; FAIL="${RD}✗${RS}"

# ── Header ──────────────────────────────────────────────────────
echo
echo -e "  ${BD}${CY}LOCALAI  ·  SETUP${RS}"
echo -e "  ${GR}We'll set up localai so you can type ${CY}llm${GR} in any terminal.${RS}"
echo

# ── Hardware ────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "  ${FAIL} python3 not found — install Xcode CLI tools: xcode-select --install"; exit 1
fi

HW_JSON="$(python3 "$SCRIPT_DIR/detect.py" --json 2>/dev/null || echo '{}')"
RAM_GB=$(echo "$HW_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ram_gb_int',16))" 2>/dev/null || echo "16")
CHIP=$(echo "$HW_JSON"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('chip','Apple Silicon'))" 2>/dev/null || echo "Apple Silicon")
IS_AS=$(echo "$HW_JSON"  | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if d.get('is_apple_silicon') else 'no')" 2>/dev/null || echo "yes")

echo -e "  ${GR}${CHIP}  ·  ${RAM_GB} GB RAM${RS}"
echo

if [ "$IS_AS" != "yes" ]; then
    echo -e "  ${FAIL} MLX requires Apple Silicon (M1 or later)."; exit 1
fi

# ── Venv ────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo -e "  ${OK} venv ready"
else
    echo -ne "  ${GR}creating venv…${RS}"
    mkdir -p "$LOCALAI_DIR"
    python3 -m venv "$VENV_DIR"
    echo -e "\r  ${OK} venv created       "
fi

# ── Dependencies ─────────────────────────────────────────────────
echo -ne "  ${GR}installing dependencies…${RS}"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV_DIR/bin/pip" install --upgrade mlx-lm psutil -q
echo -e "\r  ${OK} mlx-lm + psutil         "

# ── Model selection ──────────────────────────────────────────────
echo
echo -e "  ${BD}Models  ${GR}(${RAM_GB} GB RAM)${RS}"
echo

declare -a MODEL_LIST=(
    "phi4mini|mlx-community/Phi-4-mini-instruct-4bit|Phi-4 mini 3.8B|tiny · fast|6|safe"
    "gemma4|mlx-community/gemma-3-4b-it-4bit|Gemma 3 4B|tiny · balanced|6|safe"
    "mistral7|mlx-community/Mistral-7B-Instruct-v0.3-4bit|Mistral 7B|compact · fast|6|safe"
    "dolphin|mlx-community/Dolphin3.0-Llama3.1-8B-4bit|Dolphin 8B|compact · chat|8|safe"
    "llama8|mlx-community/Llama-3.1-8B-Instruct-4bit|Llama 3.1 8B|compact · general|8|safe"
    "q8|mlx-community/Qwen3-8B-4bit|Qwen 3 8B|compact · reasoning|8|safe"
    "gemma12|mlx-community/gemma-3-12b-it-4bit|Gemma 3 12B|mid · smart|10|safe"
    "q14|mlx-community/Qwen3-14B-4bit|Qwen 3 14B|mid · reasoning|10|safe"
    "q32_3|mlx-community/Qwen3-32B-3bit|Qwen 3 32B 3bit|large · tight RAM|14|safe"
    "mistral24|mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit|Mistral 24B|large · multilingual|16|safe"
    "q32|mlx-community/Qwen3-32B-4bit|Qwen 3 32B 4bit|large · deep|20|safe"
    "llama70|mlx-community/Llama-3.3-70B-Instruct-4bit|Llama 3.3 70B|very large|40|safe"
    "hui27|huihui-ai/Huihui-Qwen3.5-27B-abliterated|Huihui 27B|unfiltered|16|unfiltered"
    "hui35|huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated|Huihui 35B|unfiltered · MoE|20|unfiltered"
)

# Pick ONE recommended model per RAM tier
BEST_KEY=""
if   [ "$RAM_GB" -ge 40 ]; then BEST_KEY="llama70"
elif [ "$RAM_GB" -ge 20 ]; then BEST_KEY="q32"
elif [ "$RAM_GB" -ge 14 ]; then BEST_KEY="q14"
elif [ "$RAM_GB" -ge 10 ]; then BEST_KEY="gemma12"
elif [ "$RAM_GB" -ge 8 ];  then BEST_KEY="dolphin"
else                             BEST_KEY="phi4mini"
fi

IDX=0
declare -a AVAILABLE_MODELS=()
for entry in "${MODEL_LIST[@]}"; do
    IFS='|' read -r mkey mid mname mdesc mram mprofile <<< "$entry"
    if [ "$mram" -le "$RAM_GB" ] && [ "$mprofile" = "safe" ]; then
        IDX=$((IDX + 1))
        AVAILABLE_MODELS+=("$entry")
        REC=""
        if [ "$mkey" = "$BEST_KEY" ]; then REC="  ${CY}← recommended${RS}"; fi
        echo -e "  ${CY}${IDX}${RS}  ${BD}${mname}${RS}  ${GR}${mdesc}${RS}${REC}"
    fi
done

echo
echo -e "  ${GR}n  skip (models download on first use when you run ${CY}llm${GR})${RS}"
echo
printf "  Pick (1–%s, n, Enter = recommended): " "$IDX"
read -r MODEL_CHOICE

MODELS_TO_DL=()
case "${MODEL_CHOICE,,}" in
    n|no|skip)
        echo -e "  ${GR}Skipping — models download automatically the first time you run ${CY}llm${GR}.${RS}"
        ;;
    "" )
        # Empty input → download the single recommended model (if it exists in the filtered list)
        for entry in "${AVAILABLE_MODELS[@]}"; do
            IFS='|' read -r mkey mid mname mdesc mram mprofile <<< "$entry"
            if [ "$mkey" = "$BEST_KEY" ]; then
                MODELS_TO_DL+=("$mid|$mname")
                break
            fi
        done
        ;;
    * )
        # Allow comma-separated list like "1,3" to pre-download multiple models
        IFS=',' read -ra NUMS <<< "$MODEL_CHOICE"
        for num in "${NUMS[@]}"; do
            num="${num// /}"
            if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#AVAILABLE_MODELS[@]}" ]; then
                entry="${AVAILABLE_MODELS[$((num - 1))]}"
                IFS='|' read -r mkey mid mname mdesc mram mprofile <<< "$entry"
                MODELS_TO_DL+=("$mid|$mname")
            fi
        done
        ;;
esac

# Download with visible progress (HF bars disabled for cleaner output)
if [ "${#MODELS_TO_DL[@]}" -gt 0 ]; then
    echo
    DL_I=0
    DL_TOTAL="${#MODELS_TO_DL[@]}"
    for item in "${MODELS_TO_DL[@]}"; do
        IFS='|' read -r mid mname <<< "$item"
        DL_I=$((DL_I + 1))
        echo -e "  ${GR}[${DL_I}/${DL_TOTAL}] Downloading ${BD}${mname}${RS}${GR}…${RS}"
        echo
        HF_HUB_DISABLE_PROGRESS_BARS=1 "$VENV_DIR/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('${mid}')
"
        echo
        echo -e "  ${OK} [${DL_I}/${DL_TOTAL}] ${mname} ready"
    done
fi

# ── Voice ────────────────────────────────────────────────────────
echo
INSTALL_VOICE=0
if [ "$VOICE_OPT" -eq 1 ]; then
    INSTALL_VOICE=1
else
    printf "  Voice push-to-talk? +500 MB  [y/N]: "
    read -r VOICE_ANSWER
    if [[ "${VOICE_ANSWER,,}" == "y" || "${VOICE_ANSWER,,}" == "yes" ]]; then
        INSTALL_VOICE=1
    fi
fi

if [ "$INSTALL_VOICE" -eq 1 ]; then
    echo -ne "  ${GR}installing mlx-whisper…${RS}"
    "$VENV_DIR/bin/pip" install --upgrade mlx-whisper sounddevice -q
    echo -e "\r  ${OK} voice ready                "
    echo -e "  ${GR}Start with:${RS} ${CY}llm --voice${RS}"
else
    echo -e "  ${GR}Voice skipped.  (kör ${CY}setup.sh --voice${GR} senare om du vill lägga till det)${RS}"
fi

# ── Copy py files (skip if already in place, e.g. git clone into ~/.localai) ─
if [ "$SCRIPT_DIR" != "$LOCALAI_DIR" ]; then
    cp "$SCRIPT_DIR"/*.py "$LOCALAI_DIR/" 2>/dev/null || true
fi

# ── Wrappers ────────────────────────────────────────────────────
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/llm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
LOCALAI_DIR="$HOME/.localai"
VENV_PY="$LOCALAI_DIR/venv/bin/python"
CHAT_APP="$LOCALAI_DIR/chat.py"
if [ ! -x "$VENV_PY" ]; then echo "Missing venv. Run: ~/.localai/setup.sh"; exit 1; fi
if [ ! -f "$CHAT_APP" ]; then echo "Missing chat.py"; exit 1; fi
exec "$VENV_PY" "$CHAT_APP" "$@"
EOF

cat > "$BIN_DIR/llm-stop" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PID_FILE="$HOME/.localai/llm.pid"
if [ ! -f "$PID_FILE" ]; then echo "No active session."; exit 0; fi
pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "${pid:-}" ]; then rm -f "$PID_FILE"; exit 0; fi
if ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; echo "Stale PID removed."; exit 0; fi
cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
if ! printf '%s' "$cmd" | grep -q ".localai/chat.py"; then
    echo "PID $pid is not llm. Refusing."; exit 1
fi
kill "$pid" 2>/dev/null || true
sleep 0.4
kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Stopped (PID $pid)."
EOF

chmod +x "$BIN_DIR/llm" "$BIN_DIR/llm-stop"

# ── zshrc ────────────────────────────────────────────────────────
touch "$ZSHRC"
tmp="$(mktemp)"
awk -v s="$MARKER_START" -v e="$MARKER_END" '
    $0 == s { skip=1; next }
    $0 == e { skip=0; next }
    !skip { print }
' "$ZSHRC" > "$tmp"
mv "$tmp" "$ZSHRC"

{
    echo; echo "$MARKER_START"
    echo 'llm() { "$HOME/bin/llm" "$@"; }'
    echo 'llm-stop() { "$HOME/bin/llm-stop" "$@"; }'
    echo 'alias llmstop="llm-stop"'
    echo "$MARKER_END"
} >> "$ZSHRC"

# ── Done ─────────────────────────────────────────────────────────
echo
echo -e "  ${OK} ${BD}Done.${RS}  Open a new tab, type ${CY}llm${RS} and start chatting."
echo -e "  ${GR}If ${CY}llm${GR} is not found, run:${RS} ${CY}source ~/.zshrc${RS}"
echo
