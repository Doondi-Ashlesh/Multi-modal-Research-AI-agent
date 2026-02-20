"""Research tools for the agent."""
from .documents import load_document, summarize_document
from .search import web_search

__all__ = ["load_document", "summarize_document", "web_search"]
