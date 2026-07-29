"""
loaders.py
----------
Single responsibility: load raw PDF documents from disk into a list of
LangChain `Document` objects. Supports any number of PDFs dropped into
the source directory — no chunking, embedding, or storage here.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


def load_pdfs(source_dir: str | Path | None = None) -> list[Document]:
    """
    Load every PDF found under `source_dir` (defaults to `settings.raw_data_dir`).

    Each page of each PDF becomes one `Document`. Any number of PDFs may
    be present — they are all loaded and later distinguished by their
    `source` (filename) metadata.

    Returns
    -------
    list[Document]
        One Document per PDF page, across all PDFs found. Empty list if
        the directory has no PDFs.
    """
    source_dir = Path(source_dir) if source_dir else settings.raw_data_dir

    if not source_dir.exists():
        logger.warning("Raw data directory does not exist: %s", source_dir)
        return []

    pdf_files = sorted(source_dir.glob(settings.pdf_glob_pattern))
    if not pdf_files:
        logger.warning("No PDF files found in %s", source_dir)
        return []

    logger.info(
        "Loading %d PDF file(s) from %s: %s",
        len(pdf_files),
        source_dir,
        [f.name for f in pdf_files],
    )

    loader = DirectoryLoader(
        str(source_dir),
        glob=settings.pdf_glob_pattern,
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()

    logger.info(
        "Loaded %d page-level document(s) across %d file(s)",
        len(documents),
        len(pdf_files),
    )
    return documents
