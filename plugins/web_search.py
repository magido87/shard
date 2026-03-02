"""Web Search plugin — DuckDuckGo search from chat."""

PLUGIN_NAME    = "web_search"
PLUGIN_LABEL   = "Web Search"
PLUGIN_DESC    = "Search the web via DuckDuckGo from chat"
PLUGIN_DEPS    = ["duckduckgo-search"]
PLUGIN_VERSION = "1.0"


def setup() -> bool:
    try:
        from duckduckgo_search import DDGS  # noqa: F401
        return True
    except ImportError:
        return False


def on_command(cmd: str, context: dict) -> bool:
    """Handle /search <query> command."""
    if not cmd.startswith("/search "):
        return False

    query = cmd[8:].strip()
    if not query:
        print("  Usage: /search <query>")
        return True

    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=5))
        if results:
            print(f"\n  Search results for: {query}\n")
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body  = r.get("body", "")[:150]
                href  = r.get("href", "")
                print(f"  {i}. {title}")
                print(f"     {body}")
                if href:
                    print(f"     {href}")
                print()
        else:
            print("  No results found.\n")
    except Exception as e:
        print(f"  Search error: {e}\n")

    return True


def on_query(user_input: str, context: dict) -> str | None:
    """If query starts with 'search:', prepend web results to context."""
    if not user_input.lower().startswith("search:"):
        return None

    query = user_input[7:].strip()
    if not query:
        return None

    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=3))
        if results:
            context_text = "\n".join(
                f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
            )
            return (
                f"Based on these web search results:\n{context_text}\n\n"
                f"Answer this question: {query}"
            )
    except Exception:
        pass
    return None
