import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import VAULT_WIKI_PATH

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
COLLECTION_NAME = "wiki_notes"
CHUNK_WORDS = 400   # target words per chunk
OVERLAP_WORDS = 50  # overlap between consecutive chunks
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chromadb handles embedding internally via this function
_ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    step = max(1, CHUNK_WORDS - OVERLAP_WORDS)
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + CHUNK_WORDS]))
        start += step
    return chunks


def build_index() -> int:
    """
    Incrementally sync the chromadb vector index with VAULT_WIKI_PATH.

    - Skips notes whose mtime hasn't changed (already indexed).
    - Removes embeddings for notes that were deleted from the vault.
    - Returns the number of files that were (re-)indexed.
    """
    collection = _get_collection()

    # Discover which sources are already indexed
    all_indexed = collection.get(include=["metadatas"])
    indexed_sources: dict[str, float] = {}  # filename -> mtime stored in index
    for meta in (all_indexed.get("metadatas") or []):
        if meta and "source" in meta:
            src = meta["source"]
            # Keep the highest mtime seen for this source (all chunks share the same value)
            if src not in indexed_sources or meta["mtime"] > indexed_sources[src]:
                indexed_sources[src] = meta["mtime"]

    # Discover current vault files (exclude INDEX.md)
    md_files = [
        f for f in os.listdir(VAULT_WIKI_PATH)
        if f.endswith(".md") and f != "INDEX.md"
    ]
    current_files = set(md_files)

    # Purge embeddings for notes that no longer exist on disk
    for dead in set(indexed_sources) - current_files:
        dead_ids = collection.get(where={"source": dead})["ids"]
        if dead_ids:
            collection.delete(ids=dead_ids)
        print(f"[embed] Purged deleted note: {dead}")

    indexed_count = 0
    for filename in md_files:
        path = os.path.join(VAULT_WIKI_PATH, filename)
        mtime = round(os.path.getmtime(path), 3)

        # Skip if already indexed at this exact mtime
        if indexed_sources.get(filename) == mtime:
            continue

        # Remove stale chunks before re-embedding
        stale = collection.get(where={"source": filename})
        if stale["ids"]:
            collection.delete(ids=stale["ids"])

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = _chunk_text(content)
        if not chunks:
            continue

        # chromadb calls _ef internally to compute embeddings
        collection.add(
            documents=chunks,
            ids=[f"{filename}::chunk::{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": filename, "chunk_index": i, "mtime": mtime}
                for i in range(len(chunks))
            ],
        )
        indexed_count += 1

    return indexed_count


def query_index(query: str, n_results: int = 5) -> list[dict]:
    """
    Return the *n_results* chunks most semantically similar to *query*.
    Each result is {"source": str, "text": str, "distance": float}.
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, total),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"source": meta["source"], "text": doc, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
