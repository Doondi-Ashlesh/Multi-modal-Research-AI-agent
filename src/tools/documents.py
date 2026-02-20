"""Document loading and summarization tools."""
from pathlib import Path

from openai import OpenAI

from ..config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
from ..multimodal import (
    IMAGE_EXTENSIONS,
    load_pdf_text,
    load_text,
)


def _client() -> OpenAI:
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_BASE:
        kwargs["base_url"] = OPENAI_API_BASE
    return OpenAI(**kwargs)


def load_document(path: str) -> str:
    """
    Load a document from file path. Supports PDF, text, and image files.
    Returns extracted text; for images returns a placeholder (use vision in agent for analysis).
    """
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf_text(p)
    if ext in IMAGE_EXTENSIONS:
        return f"[Image file: {p.name}. Use the assistant's vision capability to analyze this image.]"
    try:
        return load_text(p)
    except Exception as e:
        return f"Error reading file: {e}"


def summarize_document(text: str, max_length: int = 1500) -> str:
    """
    Summarize long document text. Uses LLM to produce a concise summary.
    """
    if not text or len(text.strip()) < 100:
        return text
    client = _client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following document concisely, preserving key facts and conclusions. Keep the summary under {max_length} characters.\n\n---\n\n{text[:12000]}",
            }
        ],
    )
    return response.choices[0].message.content or ""
