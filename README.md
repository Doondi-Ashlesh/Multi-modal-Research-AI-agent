# Multi-Modal Research AI Agent

A research assistant that understands **text**, **images**, and **documents** (PDFs), and can use **web search** and **summarization** to answer questions with citations.

## Features

- **Multi-modal input**: Ask in plain text and attach PDFs, images (PNG, JPG, etc.), or text files.
- **Research tools**:
  - **load_document** – Load and extract text from files.
  - **summarize_document** – Summarize long documents.
  - **web_search** – Search the web for current information.
- **Agent loop**: The model can call these tools in sequence to gather and synthesize information before answering.

## Setup

1. **Clone or navigate to the project**  
   `cd "Multi modal AI agent"`

2. **Create a virtual environment (recommended)**  
   `python -m venv .venv`  
   Then activate:  
   - Windows: `.venv\Scripts\activate`  
   - macOS/Linux: `source .venv/bin/activate`

3. **Install dependencies**  
   `pip install -r requirements.txt`

4. **Configure API key**  
   Copy `.env.example` to `.env` and set your OpenAI API key (or compatible API):

   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o
   ```

   Use a vision-capable model (e.g. `gpt-4o`, `gpt-4o-mini`) to analyze images.

## Usage

**One-off query:**

```bash
python main.py "What are the main conclusions of recent papers on LLM agents?"
```

**With files (PDFs or images):**

```bash
python main.py "Summarize this paper and list the key results." -f paper.pdf
python main.py "What is in this diagram?" -f figure.png
```

**Interactive mode:**

```bash
python main.py -i
```

Then type your question and, when prompted, optional file paths (comma-separated).

## Project structure

```
Multi modal AI agent/
├── main.py              # CLI entrypoint
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── __init__.py
    ├── config.py        # Env and settings
    ├── multimodal.py    # Text + image + PDF input handling
    ├── agent.py         # Agent loop and tool orchestration
    └── tools/
        ├── __init__.py
        ├── documents.py # load_document, summarize_document
        └── search.py    # web_search (DuckDuckGo)
```

## Optional

- **Azure / other OpenAI-compatible APIs**: Set `OPENAI_API_BASE` in `.env`.
- **RAG (vector store)**: Add `chromadb` and `sentence-transformers` to `requirements.txt` and implement a “retrieve” tool that queries your index.

## License

Use and modify as you like.
