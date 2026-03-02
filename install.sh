#!/usr/bin/env bash
# localai — one-liner bootstrap
# Usage: bash <(curl -fsSL https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/install.sh)
set -euo pipefail

REPO_URL="${LOCALAI_REPO_URL:-https://github.com/YOUR_USER/YOUR_REPO.git}"
LOCALAI_DIR="$HOME/.localai"
BRANCH="${LOCALAI_BRANCH:-main}"

CY='\033[38;5;51m'; GR='\033[90m'; BD='\033[1m'
RD='\033[38;5;196m'; RS='\033[0m'
OK="${CY}✓${RS}"; FAIL="${RD}✗${RS}"

echo
echo -e "  ${BD}${CY}localai${RS}  ${GR}— offline LLM for Mac${RS}"
echo

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "  ${FAIL} localai requires macOS."; exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "  ${FAIL} localai requires Apple Silicon (M1 or later)."; exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "  ${FAIL} python3 not found. Run: xcode-select --install"; exit 1
fi

# ── Clone or update ──────────────────────────────────────────────
if [[ -d "$LOCALAI_DIR/.git" ]]; then
    echo -ne "  ${GR}Updating …${RS}"
    (cd "$LOCALAI_DIR" && git pull --quiet 2>/dev/null || true)
    echo -e "\r  ${OK} Updated                "
elif [[ -d "$LOCALAI_DIR" ]]; then
    echo -e "  ${GR}~/.localai exists but is not a git repo — using as-is.${RS}"
else
    echo -ne "  ${GR}Cloning …${RS}"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$LOCALAI_DIR"
    echo -e "\r  ${OK} Cloned              "
fi

bash "$LOCALAI_DIR/setup.sh" "$@"
