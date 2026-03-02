"""Shell Assistant plugin — quick shell command execution."""

PLUGIN_NAME    = "shell_assistant"
PLUGIN_LABEL   = "Shell Assistant"
PLUGIN_DESC    = "Suggest and run shell commands"
PLUGIN_DEPS    = []
PLUGIN_VERSION = "1.0"

import subprocess


def setup() -> bool:
    return True


def on_command(cmd: str, context: dict) -> bool:
    """Handle /sh <command>."""
    if not cmd.startswith("/sh "):
        return False

    command = cmd[4:].strip()
    if not command:
        print("  Usage: /sh <command>\n")
        return True

    print(f"\n  $ {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[:80]:
                print(f"  {line}")
            if len(lines) > 80:
                print(f"  ... ({len(lines) - 80} more lines)")
        if result.stderr:
            print(f"  {result.stderr.strip()}")
        if result.returncode != 0:
            print(f"  (exit code {result.returncode})")
        print()
    except subprocess.TimeoutExpired:
        print("  Timed out after 30s.\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    return True
