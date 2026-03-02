"""Persistent config for localai — ~/.localai/config.json"""

import json
import sys
from pathlib import Path

CONFIG_PATH = Path.home() / ".localai" / "config.json"

DEFAULTS: dict = {
    "model":            "phi4mini",
    "theme":            "ocean",
    "personality":      1,
    "temp":             0.72,
    "first_run":        True,
    "stats":            "compact",
    "log_sessions":     False,
    "privacy_mode":     False,
    "ui_mode":          "normal",
    "plugins":          [],
    "disabled_plugins": [],
}


def load() -> dict:
    """Merge config file over DEFAULTS; on error log to stderr and return defaults."""
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            for k, v in data.items():
                if k in DEFAULTS:
                    cfg[k] = v
    except Exception as e:
        print(f"[config] warning: could not load config: {e}", file=sys.stderr)
    return cfg


def save(cfg: dict) -> None:
    """Write indented JSON; no-op if privacy_mode is True."""
    if cfg.get("privacy_mode"):
        return
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {k: cfg.get(k, DEFAULTS[k]) for k in DEFAULTS}
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except Exception as e:
        print(f"[config] warning: could not save config: {e}", file=sys.stderr)
