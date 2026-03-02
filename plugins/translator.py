"""Translator plugin — auto-translate responses to another language."""

PLUGIN_NAME    = "translator"
PLUGIN_LABEL   = "Translator"
PLUGIN_DESC    = "Auto-translate responses to another language"
PLUGIN_DEPS    = []
PLUGIN_VERSION = "1.0"

_target_lang: str = ""


def setup() -> bool:
    return True


def on_command(cmd: str, context: dict) -> bool:
    """Handle /translate <language> and /translate off."""
    global _target_lang
    if not cmd.startswith("/translate"):
        return False

    arg = cmd[10:].strip()
    if not arg or arg == "off":
        _target_lang = ""
        print("  Translation: off\n")
    else:
        _target_lang = arg
        print(f"  Translation target: {_target_lang}\n")
        print(f"  Responses will be translated to {_target_lang}.\n")
    return True


def on_generate_messages(messages: list[dict], context: dict) -> list[dict] | None:
    """Inject translation instruction into system prompt."""
    if not _target_lang:
        return None

    modified = list(messages)
    inject_text = (
        f"\n\nIMPORTANT: Always respond in {_target_lang}. "
        f"Translate your response to {_target_lang} regardless of "
        f"the language of the user's question."
    )
    injected = False
    for i, msg in enumerate(modified):
        if msg["role"] == "system":
            modified[i] = {
                "role":    "system",
                "content": msg["content"] + inject_text,
            }
            injected = True
            break

    # Fallback for models without system role (e.g. Mistral)
    if not injected:
        for i, msg in enumerate(modified):
            if msg["role"] == "user":
                modified[i] = {
                    "role":    "user",
                    "content": inject_text.strip() + "\n\n" + msg["content"],
                }
                break

    return modified
