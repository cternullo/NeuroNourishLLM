import os
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import VAULT_WIKI_PATH

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chroma")
COLLECTION_NAME = "wiki_notes"
CHUNK_WORDS = 400
OVERLAP_WORDS = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_FETCH = 20   # retrieve this many from chroma before reranking
TOP_RETURN = 7   # return this many to the LLM after reranking

_ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(text: str) -> list[str]:
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


def _extract_section(text: str) -> str:
    """Return the last markdown heading found in a chunk (# / ## / ###)."""
    section = ""
    for line in text.split("\n"):
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            section = m.group(1).strip()
    return section


def _keyword_score(query: str, text: str) -> float:
    """Fraction of unique query tokens (3+ chars) present in the chunk text."""
    tokens = list({t.lower() for t in re.findall(r"\b\w{3,}\b", query)})
    if not tokens:
        return 0.0
    text_lower = text.lower()
    return sum(1 for t in tokens if t in text_lower) / len(tokens)


def build_index() -> int:
    """
    Incrementally sync the chromadb vector index with VAULT_WIKI_PATH.
    Skips notes whose mtime hasn't changed, purges deleted notes.
    Returns the number of files (re-)indexed.
    """
    collection = _get_collection()

    all_indexed = collection.get(include=["metadatas"])
    indexed_sources: dict[str, float] = {}
    for meta in (all_indexed.get("metadatas") or []):
        if meta and "source" in meta:
            src = meta["source"]
            if src not in indexed_sources or meta["mtime"] > indexed_sources[src]:
                indexed_sources[src] = meta["mtime"]

    md_files = [
        f for f in os.listdir(VAULT_WIKI_PATH)
        if f.endswith(".md") and f != "INDEX.md"
    ]
    current_files = set(md_files)

    for dead in set(indexed_sources) - current_files:
        dead_ids = collection.get(where={"source": dead})["ids"]
        if dead_ids:
            collection.delete(ids=dead_ids)
        print(f"[embed] Purged deleted note: {dead}")

    indexed_count = 0
    for filename in md_files:
        path = os.path.join(VAULT_WIKI_PATH, filename)
        mtime = round(os.path.getmtime(path), 3)

        if indexed_sources.get(filename) == mtime:
            continue

        stale = collection.get(where={"source": filename})
        if stale["ids"]:
            collection.delete(ids=stale["ids"])

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = _chunk_text(content)
        if not chunks:
            continue

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


def reindex_note(filename: str) -> None:
    """Force re-embed a single note, purging stale chunks first."""
    collection = _get_collection()
    stale = collection.get(where={"source": filename})
    if stale["ids"]:
        collection.delete(ids=stale["ids"])

    path = os.path.join(VAULT_WIKI_PATH, filename)
    if not os.path.isfile(path):
        return  # file deleted — stale chunks already removed above

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    mtime = round(os.path.getmtime(path), 3)
    chunks = _chunk_text(content)
    if not chunks:
        return

    collection.add(
        documents=chunks,
        ids=[f"{filename}::chunk::{i}" for i in range(len(chunks))],
        metadatas=[
            {"source": filename, "chunk_index": i, "mtime": mtime}
            for i in range(len(chunks))
        ],
    )


def query_index(query: str, n_results: int = TOP_RETURN) -> list[dict]:
    """
    Retrieve TOP_FETCH chunks by cosine similarity, rescore with keyword
    overlap, return the best n_results.

    Each result dict: {"source", "section", "text", "distance", "score"}
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    fetch = min(TOP_FETCH, total)
    results = collection.query(
        query_texts=[query],
        n_results=fetch,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        cosine_sim = 1.0 - dist          # cosine distance → similarity
        kw = _keyword_score(query, doc)
        combined = cosine_sim * 0.7 + kw * 0.3
        candidates.append({
            "source": meta["source"],
            "section": _extract_section(doc),
            "text": doc,
            "distance": dist,
            "score": combined,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:min(n_results, TOP_RETURN)]
