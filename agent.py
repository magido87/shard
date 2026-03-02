"""Interaction logging for localai — JSONL, opt-in only."""

import json
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path.home() / ".localai" / "logs"

_enabled: bool = False
_session_path: Path | None = None


def set_logging(enabled: bool) -> None:
    global _enabled, _session_path
    _enabled = enabled
    if enabled and _session_path is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_path = LOGS_DIR / f"session_{ts}.jsonl"


def log_interaction(
    query: str,
    answer: str,
    metadata: dict | None = None,
) -> None:
    """Append one JSONL line to the session log if logging is enabled."""
    if not _enabled or _session_path is None:
        return
    try:
        entry: dict = {
            "ts":     datetime.now().isoformat(),
            "query":  query,
            "answer": answer,
        }
        if metadata:
            entry["meta"] = metadata
        with _session_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def list_logs() -> list[Path]:
    """Return list of log file Paths, newest first."""
    if not LOGS_DIR.is_dir():
        return []
    return sorted(LOGS_DIR.glob("*.jsonl"), reverse=True)


def delete_all_logs() -> int:
    """Delete all session log files. Returns count deleted."""
    files = list_logs()
    for f in files:
        try:
            f.unlink()
        except OSError:
            pass
    return len(files)
