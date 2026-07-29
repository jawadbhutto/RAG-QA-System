"""
vector_store.py
----------------
Single responsibility: own every interaction with ChromaDB — create or
open the persistent collection, upsert chunks idempotently, run raw
similarity search, and support resetting the index.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.embeddings import get_embedding_model
from src.utils import get_logger

logger = get_logger(__name__)

_vector_db: Chroma | None = None


def get_vector_store() -> Chroma:
    """Return a singleton `Chroma` handle backed by `settings.chroma_persist_dir`."""
    global _vector_db
    if _vector_db is None:
        settings.ensure_directories()
        logger.info(
            "Opening Chroma collection '%s' at %s",
            settings.collection_name,
            settings.chroma_persist_dir,
        )
        _vector_db = Chroma(
            collection_name=settings.collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=str(settings.chroma_persist_dir),
            collection_metadata={"hnsw:space": settings.distance_metric},
        )
    return _vector_db


def _build_chunk_id(chunk: Document) -> str:
    """Build a deterministic ID so identical chunks are never duplicated."""
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", 0)
    chunk_number = chunk.metadata.get("chunk_number", 0)
    return f"{source}_{page}_{chunk_number}"


def add_chunks(chunks: list[Document]) -> int:
    """Upsert `chunks`, skipping any whose deterministic ID already exists."""
    if not chunks:
        logger.warning("add_chunks received an empty chunk list")
        return 0

    vector_db = get_vector_store()
    ids = [_build_chunk_id(chunk) for chunk in chunks]

    existing = vector_db.get(ids=ids)
    existing_ids = set(existing["ids"])

    new_docs, new_ids = [], []
    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in existing_ids:
            new_docs.append(chunk)
            new_ids.append(chunk_id)

    if new_docs:
        # Chroma caps how many docs can be added in a single call.
        # Try to read the real limit from the client; fall back to a safe default.
        try:
            max_batch = vector_db._client.get_max_batch_size()
        except Exception:
            max_batch = 5000

        for i in range(0, len(new_docs), max_batch):
            batch_docs = new_docs[i:i + max_batch]
            batch_ids = new_ids[i:i + max_batch]
            vector_db.add_documents(documents=batch_docs, ids=batch_ids)

        logger.info("Added %d new chunk(s) to the vector store", len(new_docs))
    else:
        logger.info("Vector store already up to date; no new chunks added")

    return len(new_docs)


def collection_count() -> int:
    """Return the total number of vectors currently stored."""
    return get_vector_store()._collection.count()


def similarity_search(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """
    Run a raw similarity search and return (Document, distance) pairs
    exactly as Chroma reports them — lower distance means more similar.
    """
    vector_db = get_vector_store()
    k = k or settings.top_k
    return vector_db.similarity_search_with_score(query, k=k)


def reset_vector_store() -> None:
    """
    Delete the entire collection and drop the cached handle so the next
    call to `get_vector_store()` recreates it fresh. Use this to fully
    re-index from scratch (e.g. after removing or replacing source PDFs).
    """
    global _vector_db
    vector_db = get_vector_store()
    vector_db.delete_collection()
    _vector_db = None
    logger.info("Vector store collection '%s' deleted", settings.collection_name)
