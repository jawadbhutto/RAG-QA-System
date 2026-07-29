"""
splitter.py
-----------
Single responsibility: split loaded Documents into smaller, overlapping
chunks suitable for embedding, and stamp each chunk with clean, useful
metadata (basename source, sequential chunk number, document type).

Because citations later depend on this metadata being accurate, this
module is the one place responsible for getting it right.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split `documents` into chunks of `settings.chunk_size` characters with
    `settings.chunk_overlap` overlap, then normalize their metadata.

    Returns
    -------
    list[Document]
        Chunked documents, each with metadata:
        - source: PDF filename only (no full path) — used for citations
        - chunk_number: 1-indexed position across the whole corpus
        - document_type: "pdf"
        - page / page_label: carried over from the original loader
    """
    if not documents:
        logger.warning("split_documents received an empty document list")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata.update(
            {
                "source": os.path.basename(chunk.metadata.get("source", "unknown")),
                "chunk_number": i,
                "document_type": "pdf",
            }
        )

    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
