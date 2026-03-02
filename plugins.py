"""Plugin registry, loader, and installer for localai."""

import importlib.util
import subprocess
import sys
from pathlib import Path

PLUGINS_DIR = Path.home() / ".localai" / "plugins"
VENV_PIP    = Path.home() / ".localai" / "venv" / "bin" / "pip"

# ── Plugin Registry ──────────────────────────────────────────────
REGISTRY: dict = {
    "web_search": {
        "label": "Web Search",
        "desc":  "Search the web via DuckDuckGo from chat",
        "deps":  ["duckduckgo-search"],
        "file":  "web_search.py",
    },
    "code_runner": {
        "label": "Code Runner",
        "desc":  "Execute Python/shell code from responses",
        "deps":  [],
        "file":  "code_runner.py",
    },
    "file_reader": {
        "label": "File Reader (RAG)",
        "desc":  "Load local files into chat context",
        "deps":  [],
        "file":  "file_reader.py",
    },
    "tts": {
        "label": "Text-to-Speech",
        "desc":  "Read responses aloud via macOS say",
        "deps":  [],
        "file":  "tts.py",
    },
    "shell_assistant": {
        "label": "Shell Assistant",
        "desc":  "Suggest and run shell commands",
        "deps":  [],
        "file":  "shell_assistant.py",
    },
    "summarizer": {
        "label": "Summarizer",
        "desc":  "Summarize long text or fetch URLs",
        "deps":  ["requests"],
        "file":  "summarizer.py",
    },
    "translator": {
        "label": "Translator",
        "desc":  "Auto-translate responses to another language",
        "deps":  [],
        "file":  "translator.py",
    },
    "clipboard": {
        "label": "Clipboard",
        "desc":  "Copy last response to clipboard",
        "deps":  [],
        "file":  "clipboard.py",
    },
}

# ── Loaded plugin modules ────────────────────────────────────────
_loaded: dict = {}   # name -> module
_last_response: str = ""  # for clipboard plugin


def installed_plugins(cfg: dict) -> list[str]:
    """Return list of installed plugin names from config."""
    return cfg.get("plugins", [])


def enabled_plugins(cfg: dict) -> list[str]:
    """Return list of enabled (installed + active) plugin names."""
    disabled = cfg.get("disabled_plugins", [])
    return [p for p in installed_plugins(cfg) if p not in disabled]


def install_plugin(name: str, cfg: dict) -> tuple[bool, str]:
    """Install a plugin's dependencies and add to config."""
    if name not in REGISTRY:
        return False, f"Unknown plugin: {name}"

    entry = REGISTRY[name]

    # Install pip dependencies
    if entry["deps"]:
        try:
            subprocess.run(
                [str(VENV_PIP), "install", "-q"] + entry["deps"],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            return False, f"Failed to install deps: {e.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return False, "Install timed out"

    # Add to config
    plugins = cfg.get("plugins", [])
    if name not in plugins:
        plugins.append(name)
        cfg["plugins"] = plugins

    return True, f"Installed {entry['label']}"


def uninstall_plugin(name: str, cfg: dict) -> tuple[bool, str]:
    """Remove plugin from config (does NOT uninstall pip packages)."""
    plugins = cfg.get("plugins", [])
    if name in plugins:
        plugins.remove(name)
        cfg["plugins"] = plugins
    disabled = cfg.get("disabled_plugins", [])
    if name in disabled:
        disabled.remove(name)
        cfg["disabled_plugins"] = disabled

    if name in _loaded:
        del _loaded[name]

    label = REGISTRY.get(name, {}).get("label", name)
    return True, f"Removed {label}"


def toggle_plugin(name: str, cfg: dict) -> bool:
    """Toggle a plugin on/off. Returns new enabled state."""
    disabled = cfg.get("disabled_plugins", [])
    if name in disabled:
        disabled.remove(name)
        enabled = True
    else:
        disabled.append(name)
        enabled = False
    cfg["disabled_plugins"] = disabled
    return enabled


def load_plugins(cfg: dict) -> None:
    """Load all enabled plugins into memory."""
    global _loaded
    _loaded = {}

    for name in enabled_plugins(cfg):
        if name not in REGISTRY:
            continue

        plugin_file = _find_plugin_file(name)
        if not plugin_file:
            continue

        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{name}", plugin_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if hasattr(mod, "setup") and not mod.setup():
                continue

            _loaded[name] = mod
        except Exception as e:
            print(f"  [plugin] warning: failed to load {name}: {e}", file=sys.stderr)


def _find_plugin_file(name: str) -> Path | None:
    """Locate a plugin file — check deployed dir first, then source dir."""
    entry = REGISTRY.get(name)
    if not entry:
        return None

    deployed = PLUGINS_DIR / entry["file"]
    if deployed.exists():
        return deployed

    src_dir = Path(__file__).parent / "plugins"
    src = src_dir / entry["file"]
    if src.exists():
        return src

    return None


def get_loaded() -> dict:
    """Return dict of currently loaded plugin modules."""
    return dict(_loaded)


def set_last_response(text: str) -> None:
    """Store the last model response (used by clipboard plugin)."""
    global _last_response
    _last_response = text


def get_last_response() -> str:
    """Get the last model response."""
    return _last_response


# ── Hook dispatchers ─────────────────────────────────────────────

def run_on_query(user_input: str, context: dict) -> str:
    """Run all on_query hooks. Returns (possibly modified) user input."""
    result = user_input
    for _name, mod in _loaded.items():
        if hasattr(mod, "on_query"):
            try:
                modified = mod.on_query(result, context)
                if modified is not None:
                    result = modified
            except Exception:
                pass
    return result


def run_on_response(response: str, query: str, context: dict) -> str:
    """Run all on_response hooks. Returns (possibly modified) response."""
    result = response
    for _name, mod in _loaded.items():
        if hasattr(mod, "on_response"):
            try:
                modified = mod.on_response(result, query, context)
                if modified is not None:
                    result = modified
            except Exception:
                pass
    return result


def run_on_command(cmd: str, context: dict) -> bool:
    """Run on_command hooks. Returns True if any plugin handled the command."""
    for _name, mod in _loaded.items():
        if hasattr(mod, "on_command"):
            try:
                if mod.on_command(cmd, context):
                    return True
            except Exception:
                pass
    return False


def run_on_generate_messages(messages: list[dict], context: dict) -> list[dict]:
    """Run on_generate_messages hooks. Returns (possibly modified) messages."""
    result = messages
    for _name, mod in _loaded.items():
        if hasattr(mod, "on_generate_messages"):
            try:
                modified = mod.on_generate_messages(result, context)
                if modified is not None:
                    result = modified
            except Exception:
                pass
    return result
