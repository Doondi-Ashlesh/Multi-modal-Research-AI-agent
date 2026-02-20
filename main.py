#!/usr/bin/env python3
"""CLI entrypoint for the multi-modal research AI agent."""
import argparse
from pathlib import Path

from src.agent import run_research
from src.config import OPENAI_API_KEY


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-modal research AI agent: ask questions, attach PDFs/images, get researched answers."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Your research question or instruction.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        default=[],
        help="Paths to PDF, image, or text files to include.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for queries).",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("Error: Set OPENAI_API_KEY in .env or environment.")
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
