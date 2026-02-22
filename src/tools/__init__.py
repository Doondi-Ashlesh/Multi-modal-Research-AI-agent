"""Research tools for the agent."""
from .academic_papers import search_academic_papers
from .documents import load_document, summarize_document
from .rag import index_document, index_directory, retrieve_from_knowledge_base
from .search import web_search

__all__ = [
    "load_document",
    "summarize_document",
    "web_search",
    "search_academic_papers",
    "retrieve_from_knowledge_base",
    "index_document",
    "index_directory",
]
