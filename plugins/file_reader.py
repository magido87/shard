"""File Reader plugin — load local files into chat context."""

PLUGIN_NAME    = "file_reader"
PLUGIN_LABEL   = "File Reader (RAG)"
PLUGIN_DESC    = "Load local files into chat context"
PLUGIN_DEPS    = []
PLUGIN_VERSION = "1.0"

import os

_file_context: list[dict] = []


def setup() -> bool:
    return True


def on_command(cmd: str, context: dict) -> bool:
    """Handle /read <path>, /files, /clearfiles."""
    if cmd.startswith("/read "):
        path = cmd[6:].strip()
        path = os.path.expanduser(path)
        real = os.path.realpath(path)
        BLOCKED = [os.path.expanduser("~/.ssh"), os.path.expanduser("~/.localai")]
        if any(real.startswith(b) for b in BLOCKED):
            print("  Blocked: sensitive path.\n")
            return True
        if not os.path.isfile(path):
            print(f"  File not found: {path}\n")
            return True
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated at 10k chars)"
            _file_context.append({"path": path, "content": content})
            print(f"  Loaded: {path} ({len(content)} chars)\n")
        except Exception as e:
            print(f"  Error reading file: {e}\n")
        return True

    if cmd == "/files":
        if _file_context:
            print()
            for f in _file_context:
                print(f"  - {f['path']} ({len(f['content'])} chars)")
            print()
        else:
            print("  No files loaded. Use: /read <path>\n")
        return True

    if cmd == "/clearfiles":
        count = len(_file_context)
        _file_context.clear()
        print(f"  Cleared {count} file(s) from context.\n")
        return True

    return False


def on_generate_messages(messages: list[dict], context: dict) -> list[dict] | None:
    """Inject loaded file contents into the system message."""
    if not _file_context:
        return None

    file_text = "\n\n".join(
        f"=== File: {f['path']} ===\n{f['content']}" for f in _file_context
    )

    modified = list(messages)
    injected = False
    for i, msg in enumerate(modified):
        if msg["role"] == "system":
            modified[i] = {
                "role":    "system",
                "content": msg["content"] + f"\n\nContext files:\n{file_text}",
            }
            injected = True
            break

    # Fallback for models without system role (e.g. Mistral)
    if not injected:
        for i, msg in enumerate(modified):
            if msg["role"] == "user":
                modified[i] = {
                    "role":    "user",
                    "content": f"Context files:\n{file_text}\n\n" + msg["content"],
                }
                break

    return modified
