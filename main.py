#!/usr/bin/env python3
"""CLI entrypoint for the multi-modal research AI agent."""
import argparse
import os
from pathlib import Path

# Suppress TensorFlow oneDNN/protobuf noise if RAG (sentence-transformers) is used later
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from src.agent import run_research
from src.config import OPENAI_API_KEY, OPENAI_API_BASE, get_api_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-modal research AI agent: ask questions, attach PDFs/images, get researched answers."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Your research question or instruction (or use 'index' to index documents for RAG).",
    )
    parser.add_argument(
        "index_path",
        nargs="?",
        default="",
        help="For 'index' command: file or directory path to index. Otherwise ignored.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        default=[],
        help="Paths to PDF, image, or text files to include (or path for 'index' if index_path not used).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for queries).",
    )
    args = parser.parse_args()

    # RAG: index documents into the knowledge base (lazy-import to avoid loading TensorFlow)
    if args.query and args.query.strip().lower() == "index":
        from src.tools.rag import index_document, index_directory
        path = (args.index_path or (args.files or [""])[0] or ".").strip()
        p = Path(path)
        if p.is_dir():
            print(index_directory(str(p)))
        else:
            print(index_document(str(p)))
        return

    if not get_api_key():
        env_path = Path(__file__).resolve().parent / ".env"
        print("Error: Configure LLM in .env. Options: (1) OPENAI_API_KEY=sk-... for OpenAI, or (2) free LLM: OPENAI_API_BASE=http://localhost:11434/v1 and OPENAI_MODEL=llama3.2 (see README).")
        if not env_path.exists():
            print("  Create .env from .env.example.")
        return

    if args.interactive:
        print("Multi-modal Research Agent (interactive). Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            files_input = input("Files (comma-separated paths, or Enter to skip): ").strip()
            file_paths = [p.strip() for p in files_input.split(",") if p.strip()]
            print("\nThinking...\n")
            answer = run_research(query, file_paths=file_paths or None)
            print("Agent:", answer, "\n")
        return

    query = args.query.strip()
    if not query:
        parser.print_help()
        return

    file_paths = args.files
    print("Thinking...\n")
    answer = run_research(query, file_paths=file_paths or None)
    print(answer)


if __name__ == "__main__":
    main()
