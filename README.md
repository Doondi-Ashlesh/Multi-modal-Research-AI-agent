# Multi-Modal Research AI Agent

A research assistant that understands **text**, **images**, and **documents** (PDFs), and can use **web search** and **summarization** to answer questions with citations.

## Features

- **Multi-modal input**: Ask in plain text and attach PDFs, images (PNG, JPG, etc.), or text files.
- **Research tools**:
  - **load_document** – Load and extract text from files.
  - **summarize_document** – Summarize long documents.
  - **web_search** – Search the web for current information.
  - **search_academic_papers** – Find papers by topic (Semantic Scholar + arXiv).
  - **retrieve_from_knowledge_base** – RAG: search a pre-indexed corpus (Chroma + sentence-transformers).
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

4. **Configure your LLM**  
   Copy `.env.example` to `.env`. Choose one:

   - **Free (Ollama, local):** Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, then in `.env` set:
     ```
     OPENAI_API_BASE=http://localhost:11434/v1
     OPENAI_MODEL=llama3.2
     ```
     Leave `OPENAI_API_KEY` empty. The agent uses tool-calling; Ollama’s `llama3.2` and many other models support it.

   - **Paid (OpenAI):** Set `OPENAI_API_KEY=sk-...` and `OPENAI_MODEL=gpt-4o` (or another model). Use a vision-capable model if you attach images.

**Notes**

- **PATH warning:** If pip warns about script location, you can add the venv `Scripts` folder to your PATH or run `pip install --no-warn-script-location -r requirements.txt`.
- **protobuf / TensorFlow:** If you see a dependency conflict with `tensorflow-intel` and `protobuf`, use a **dedicated venv** for this project (recommended). That keeps this project’s dependencies separate so TensorFlow elsewhere is unaffected.

## Usage

**One-off query:**

```bash
python main.py "What are the main conclusions of recent papers on LLM agents?"
```

**Find papers on a topic** (agent uses Semantic Scholar + arXiv):

```bash
python main.py "Find relevant papers on transformer architectures for vision"
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

**RAG (knowledge base):** Index PDFs or text files, then ask questions over them. The agent will call `retrieve_from_knowledge_base` when relevant.

```bash
# Index a file or a directory of PDFs/.txt files
python main.py index paper.pdf
python main.py index -f ./papers

# Then ask questions; the agent can search the indexed docs
python main.py "What did we conclude about X in our indexed documents?"
```

Index data is stored under `data/chroma/`.

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
        ├── academic_papers.py # search_academic_papers (Semantic Scholar, arXiv)
        ├── documents.py       # load_document, summarize_document
        ├── rag.py             # RAG: index + retrieve_from_knowledge_base
        └── search.py          # web_search (DuckDuckGo)
```

