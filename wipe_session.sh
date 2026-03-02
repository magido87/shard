#!/usr/bin/env bash
set -euo pipefail

# Securely wipe terminal/session artefacts that could hold chat text.

# Clear terminal scrollback (iTerm2 + ANSI fallback).
printf '\033]1337;ClearScrollback\a\033[3J\033[2J\033[H' || true

# Files to overwrite then delete (if they exist).
files=(
  "$HOME/.zsh_history"
  "$HOME/.zhistory"
  "$HOME/Library/Application Support/iTerm2/chatdb.sqlite"
  "$HOME/Library/Application Support/iTerm2/chatdb.sqlite-wal"
  "$HOME/Library/Application Support/iTerm2/chatdb.sqlite-shm"
  "$HOME/Library/Application Support/iTerm2/SavedState/restorable-state.sqlite"
  "$HOME/Library/Application Support/iTerm2/SavedState/restorable-state.sqlite-wal"
  "$HOME/Library/Application Support/iTerm2/SavedState/restorable-state.sqlite-shm"
)

# Directories to purge recursively.
dirs=(
  "$HOME/Library/Application Support/iTerm2/parsers"
  "$HOME/Library/Application Support/iTerm2/SavedState"
  "$HOME/Library/Application Support/iTerm2/Saved Output"
)

python_overwrite() {
  local f=$1
  local size
  size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "${size:-0}" -le 0 ]; then
    rm -f "$f"
    return
  fi
  TARGET_FILE="$f" TARGET_SIZE="$size" python3 - <<'PY'
import os
path = os.environ['TARGET_FILE']
size = int(os.environ['TARGET_SIZE'])
chunk = 1024 * 1024
with open(path, 'r+b', buffering=0) as fh:
    remaining = size
    zero = b"\x00" * chunk
    while remaining > 0:
        n = chunk if remaining >= chunk else remaining
        fh.write(zero[:n])
        remaining -= n
    fh.flush()
    fh.seek(0)
    remaining = size
    while remaining > 0:
        n = chunk if remaining >= chunk else remaining
        fh.write(os.urandom(n))
        remaining -= n
    fh.flush()
os.remove(path)
PY
}

for f in "${files[@]}"; do
  [ -f "$f" ] || continue
  python_overwrite "$f" || true
done

for d in "${dirs[@]}"; do
  [ -d "$d" ] || continue
  rm -rf "$d" || true
done

exit 0
