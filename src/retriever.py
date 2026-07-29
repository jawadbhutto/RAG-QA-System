"""
retriever.py
------------
Single responsibility: turn a standalone question into a ranked,
relevance-filtered list of retrieved chunks. This is the boundary
between "raw vector search" (vector_store.py) and "what counts as
relevant enough to answer with" (this module).

Downstream (qa_chain.py) never talks to Chroma directly — it only ever
sees `RetrievedChunk` objects from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import settings
from src.utils import get_logger
from src.vector_store import similarity_search

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk plus everything needed to cite it."""

    text: str
    source: str
    page: int | str
    chunk_number: int
    similarity: float

    @classmethod
    def from_document(cls, document: Document, similarity: float) -> "RetrievedChunk":
        metadata = document.metadata
        return cls(
            text=document.page_content,
            source=metadata.get("source", "unknown"),
            page=metadata.get("page_label", metadata.get("page", "?")),
            chunk_number=metadata.get("chunk_number", -1),
            similarity=similarity,
        )


def _distance_to_similarity(distance: float) -> float:
    """
    Convert Chroma's cosine *distance* into an intuitive [0, 1]
    *similarity* score (1.0 = identical, 0.0 = unrelated).
    """
    return max(0.0, 1.0 - distance)


def retrieve(
    question: str,
    k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for `question`.

    Returns
    -------
    list[RetrievedChunk]
        Relevant chunks, ordered best-first. Empty list if nothing
        clears `threshold` — the caller (qa_chain.py) is responsible
        for turning that into a "not found" answer.
    """
    question = (question or "").strip()
    if not question:
        logger.warning("Empty question passed to retrieve()")
        return []

    k = k or settings.top_k
    threshold = settings.relevance_threshold if threshold is None else threshold

    raw_results = similarity_search(question, k=k)
    scored = [
        RetrievedChunk.from_document(doc, _distance_to_similarity(dist))
        for doc, dist in raw_results
    ]
    relevant = [chunk for chunk in scored if chunk.similarity >= threshold]

    if not relevant:
        logger.info(
            "No chunks cleared relevance threshold %.2f for question: %r",
            threshold,
            question,
        )
        return []

    logger.info("Retrieved %d relevant chunk(s) for question: %r", len(relevant), question)
    return relevant
