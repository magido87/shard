"""Clipboard plugin — copy last response to clipboard."""

PLUGIN_NAME    = "clipboard"
PLUGIN_LABEL   = "Clipboard"
PLUGIN_DESC    = "Copy last response to clipboard"
PLUGIN_DEPS    = []
PLUGIN_VERSION = "1.0"

import shutil
import subprocess


def setup() -> bool:
    return shutil.which("pbcopy") is not None


def on_command(cmd: str, context: dict) -> bool:
    """Handle /copy command."""
    if cmd != "/copy":
        return False

    try:
        import plugins
        last = plugins.get_last_response()
    except Exception:
        last = ""

    if not last:
        print("  No response to copy.\n")
        return True

    try:
        proc = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=last.encode("utf-8"))
        chars = len(last)
        print(f"  Copied to clipboard ({chars} chars).\n")
    except Exception as e:
        print(f"  Copy failed: {e}\n")

    return True
