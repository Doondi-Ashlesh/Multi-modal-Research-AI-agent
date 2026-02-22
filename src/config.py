"""Configuration from environment."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API (OpenAI or compatible; for local/free LLMs set OPENAI_API_BASE and optionally skip key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# For local endpoints (e.g. Ollama) that don't require a key, the client still needs a placeholder
def get_api_key() -> str:
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    if OPENAI_API_BASE and "localhost" in OPENAI_API_BASE:
        return "ollama"  # placeholder for local LLMs that ignore the key
    return ""

# Agent
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
