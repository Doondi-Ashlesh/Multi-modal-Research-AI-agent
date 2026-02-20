"""RAG tool: retrieve from a pre-indexed knowledge base (Chroma + sentence-transformers)."""
from pathlib import Path
import re

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False

from ..config import PROJECT_ROOT
from ..multimodal import load_pdf_text, load_text, IMAGE_EXTENSIONS

# Default collection and persist directory
RAG_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma"
RAG_COLLECTION_NAME = "research_kb"

# Chunk size for indexing
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - CHUNK_OVERLAP
    return chunks


def _get_client_and_embedder():
    if not HAS_RAG_DEPS:
        raise RuntimeError(
            "RAG dependencies missing. Install with: pip install chromadb sentence-transformers"
        )
    RAG_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(RAG_PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model


def get_collection():
    """Get or create the default Chroma collection."""
    client, model = _get_client_and_embedder()
    return client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={"description": "Research knowledge base"},
    ), model


def index_document(path: str) -> str:
    """
    Index a single document (PDF or .txt) into the knowledge base.
    Returns a status message.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    ext = p.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "Images cannot be indexed for text retrieval. Use PDF or .txt files."
    if ext == ".pdf":
        text = load_pdf_text(p)
    else:
        try:
            text = load_text(p)
        except Exception as e:
            return f"Error reading file: {e}"
    if not text.strip():
        return f"No text extracted from {path}."
    try:
        coll, model = get_collection()
        chunks = _chunk_text(text)
        if not chunks:
            return f"No chunks from {path}."
        ids = [f"{p.name}_{i}" for i in range(len(chunks))]
        embeddings = model.encode(chunks).tolist()
        coll.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=[{"source": path}] * len(chunks))
        return f"Indexed {len(chunks)} chunks from {path}."
    except Exception as e:
        return f"Indexing error: {e}"


def index_directory(dir_path: str) -> str:
    """Index all PDF and .txt files in a directory (non-recursive)."""
    d = Path(dir_path)
    if not d.is_dir():
        return f"Error: Not a directory: {dir_path}"
    count = 0
    errors = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".pdf", ".txt"):
            result = index_document(str(p))
            if result.startswith("Indexed"):
                count += 1
            else:
                errors.append(f"{p.name}: {result}")
    msg = f"Indexed {count} file(s) from {dir_path}."
    if errors:
        msg += " " + "; ".join(errors[:5])
    return msg


def retrieve_from_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the pre-indexed knowledge base for passages relevant to the query.
    Use when the user has added documents to the knowledge base and you need to find relevant context.
    """
    if not HAS_RAG_DEPS:
        return (
            "RAG is not available. Install with: pip install chromadb sentence-transformers"
        )
    try:
        coll, model = get_collection()
        n = coll.count()
        if n == 0:
            return "The knowledge base is empty. Index documents first with: python main.py index <path_or_dir>"
        query_embedding = model.encode([query]).tolist()
        results = coll.query(query_embeddings=query_embedding, n_results=min(n_results, n), include=["documents", "metadatas"])
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        if not docs:
            return "No relevant passages found in the knowledge base."
        lines = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas or [{}] * len(docs)), 1):
            source = meta.get("source", "unknown")
            lines.append(f"[{i}] (from {source})\n{doc}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"Retrieval error: {e}"
