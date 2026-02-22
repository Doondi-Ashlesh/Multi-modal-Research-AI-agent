"""Multi-modal research agent: reasoning loop with tools."""
from typing import Any

from openai import OpenAI

from .config import get_api_key, OPENAI_API_BASE, OPENAI_MODEL, MAX_AGENT_STEPS
from .multimodal import build_message_content
from .tools import (
    load_document,
    retrieve_from_knowledge_base,
    search_academic_papers,
    summarize_document,
    web_search,
)


# Tool implementations for the agent
TOOL_IMPLEMENTATIONS = {
    "load_document": load_document,
    "summarize_document": summarize_document,
    "web_search": web_search,
    "search_academic_papers": search_academic_papers,
    "retrieve_from_knowledge_base": retrieve_from_knowledge_base,
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_document",
            "description": "Load and extract text from a document file. Supports PDF, .txt, and image paths (images return a note to use vision).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Summarize long document text concisely, preserving key facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The document text to summarize."},
                    "max_length": {"type": "integer", "description": "Max summary length in characters.", "default": 1500},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for recent facts, papers, or sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Max number of results.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_academic_papers",
            "description": "Search for academic papers by topic (Semantic Scholar and arXiv). Use when the user wants to find relevant papers, do a literature search, or discover research on a topic. Prefer this over web_search for paper discovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or research question to search for."},
                    "max_results": {"type": "integer", "description": "Maximum number of papers to return.", "default": 10},
                    "source": {"type": "string", "description": "Source: 'both' (default), 'semantic_scholar', or 'arxiv'.", "default": "both"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_knowledge_base",
            "description": "Search the pre-indexed knowledge base for relevant passages. Use when the user has indexed documents and you need to find context from their private docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for semantic retrieval."},
                    "n_results": {"type": "integer", "description": "Number of passages to return.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


def _client() -> OpenAI:
    kwargs = {"api_key": get_api_key()}
    if OPENAI_API_BASE:
        kwargs["base_url"] = OPENAI_API_BASE
    return OpenAI(**kwargs)


def _run_tool(name: str, arguments: dict[str, Any]) -> str:
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if not impl:
        return f"Unknown tool: {name}"
    try:
        result = impl(**arguments)
        return str(result) if result is not None else ""
    except Exception as e:
        return f"Tool error: {e}"


def run_research(
    user_text: str,
    file_paths: list[str] | None = None,
    system_prompt: str | None = None,
) -> str:
    """
    Run the multi-modal research agent on the given user input and optional files.
    Returns the final assistant reply.
    """
    from pathlib import Path

    client = _client()
    default_system = """You are a multi-modal research assistant. You can:
- Answer questions using your knowledge.
- Load and analyze documents (PDF, text, images) when the user provides file paths or asks about files.
- Summarize long documents.
- Search the web for current information when needed.
- Search for academic papers (Semantic Scholar, arXiv) when the user wants literature or paper discovery.
- Search the pre-indexed knowledge base (RAG) when the user has indexed documents.

Use tools when they would improve your answer. Prefer search_academic_papers over web_search for finding papers. Cite sources when you use web_search or document content. If the user attaches images or PDFs, analyze them and respond accordingly. Be concise but thorough."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt or default_system},
    ]

    content = build_message_content(user_text, [Path(p) for p in (file_paths or [])])
    messages.append({"role": "user", "content": content})

    for _ in range(MAX_AGENT_STEPS):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": getattr(msg, "tool_calls", None),
        })

        if not getattr(msg, "tool_calls", None):
            return (msg.content or "").strip()

        for tc in msg.tool_calls:
            name = tc.function.name
            import json
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return (messages[-1].get("content") or "Max steps reached.").strip()
