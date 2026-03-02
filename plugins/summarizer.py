"""Summarizer plugin — summarize text or fetch URL content."""

PLUGIN_NAME    = "summarizer"
PLUGIN_LABEL   = "Summarizer"
PLUGIN_DESC    = "Summarize long text or fetch URLs"
PLUGIN_DEPS    = ["requests"]
PLUGIN_VERSION = "1.0"


def setup() -> bool:
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


def on_command(cmd: str, context: dict) -> bool:
    """Handle /summarize <url|text>."""
    if not cmd.startswith("/summarize "):
        return False

    target = cmd[11:].strip()
    if not target:
        print("  Usage: /summarize <url or text>\n")
        return True

    if target.startswith("http://") or target.startswith("https://"):
        _summarize_url(target)
    else:
        # For plain text, inject it as context for the model
        print(f"  Text loaded ({len(target)} chars). Ask a question about it.\n")

    return True


def on_query(user_input: str, context: dict) -> str | None:
    """If query starts with 'summarize:', fetch URL and ask model to summarize."""
    if not user_input.lower().startswith("summarize:"):
        return None

    target = user_input[10:].strip()
    if target.startswith("http://") or target.startswith("https://"):
        content = _fetch_url(target)
        if content:
            return (
                f"Summarize the following text concisely:\n\n{content[:5000]}"
            )
    return None


def _summarize_url(url: str) -> None:
    """Fetch and display URL content summary."""
    content = _fetch_url(url)
    if content:
        # Show first 500 chars as preview
        preview = content[:500].strip()
        if len(content) > 500:
            preview += "..."
        print(f"\n  Fetched {len(content)} chars from {url}")
        print(f"  Preview: {preview}\n")
        print("  Use 'summarize: <url>' as a prompt to get an AI summary.\n")
    else:
        print(f"  Could not fetch content from {url}\n")


def _fetch_url(url: str) -> str | None:
    """Fetch text content from a URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        print("  Only http/https URLs supported.\n")
        return None
    try:
        import requests
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            # Basic HTML to text — strip tags
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        else:
            return resp.text
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None
