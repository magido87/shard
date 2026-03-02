"""Code Runner plugin — execute Python/shell code from chat."""

PLUGIN_NAME    = "code_runner"
PLUGIN_LABEL   = "Code Runner"
PLUGIN_DESC    = "Execute Python/shell code from responses"
PLUGIN_DEPS    = []
PLUGIN_VERSION = "1.0"

import subprocess
import sys


def setup() -> bool:
    return True


def on_command(cmd: str, context: dict) -> bool:
    """Handle /run <code> and /shell <command>."""
    if cmd.startswith("/run "):
        code = cmd[5:].strip()
        if not code:
            print("  Usage: /run <python code>")
            return True
        _run_python(code)
        return True
    if cmd.startswith("/shell "):
        command = cmd[7:].strip()
        if not command:
            print("  Usage: /shell <command>")
            return True
        _run_shell(command)
        return True
    return False


def _run_python(code: str) -> None:
    """Run Python code in a subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            print()
            for line in lines[:50]:
                print(f"  {line}")
            if len(lines) > 50:
                print(f"  ... ({len(lines) - 50} more lines)")
            print()
        if result.stderr:
            print(f"\n  Error:\n  {result.stderr.strip()}\n")
        if not result.stdout and not result.stderr:
            print(f"  (no output, exit code {result.returncode})\n")
    except subprocess.TimeoutExpired:
        print("  Timed out after 30s.\n")
    except Exception as e:
        print(f"  Error: {e}\n")


def _run_shell(command: str) -> None:
    """Run a shell command in subprocess."""
    print(f"\n  $ {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[:50]:
                print(f"  {line}")
            if len(lines) > 50:
                print(f"  ... ({len(lines) - 50} more lines)")
        if result.stderr:
            print(f"  {result.stderr.strip()}")
        print()
    except subprocess.TimeoutExpired:
        print("  Timed out after 30s.\n")
    except Exception as e:
        print(f"  Error: {e}\n")
