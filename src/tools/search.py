"""Web search tool for research."""
from typing import Any

try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for the given query. Returns a formatted string of snippets and titles.
    Requires: pip install duckduckgo-search
    """
    if not HAS_DUCKDUCKGO:
        return (
            "Web search is not available. Install with: pip install duckduckgo-search"
        )
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"
