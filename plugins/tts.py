"""Text-to-Speech plugin — read responses aloud via macOS say."""

PLUGIN_NAME    = "tts"
PLUGIN_LABEL   = "Text-to-Speech"
PLUGIN_DESC    = "Read responses aloud via macOS say"
PLUGIN_DEPS    = []
PLUGIN_VERSION = "1.0"

import shutil
import subprocess

_enabled = True


def setup() -> bool:
    return shutil.which("say") is not None


def on_command(cmd: str, context: dict) -> bool:
    """Handle /tts toggle and /say <text>."""
    global _enabled
    if cmd == "/tts":
        _enabled = not _enabled
        state = "on" if _enabled else "off"
        print(f"  TTS: {state}\n")
        return True
    if cmd.startswith("/say "):
        text = cmd[5:].strip()
        if text:
            _speak(text)
        return True
    return False


def on_response(response: str, query: str, context: dict) -> str | None:
    """Read the response aloud if TTS is enabled."""
    if not _enabled:
        return None
    _speak(response)
    return None  # don't modify the response text


def _speak(text: str) -> None:
    """Speak text via macOS say command."""
    try:
        # Strip markdown formatting for cleaner speech
        clean = text.replace("```", "").replace("`", "")
        clean = clean.replace("**", "").replace("*", "")
        clean = clean.replace("###", "").replace("##", "").replace("#", "")
        # Truncate long text
        if len(clean) > 500:
            clean = clean[:500] + "..."
        subprocess.Popen(
            ["say", clean],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
