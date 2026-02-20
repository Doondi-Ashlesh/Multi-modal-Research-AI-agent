"""Configuration from environment."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Agent
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
