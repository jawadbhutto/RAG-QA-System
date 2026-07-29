"""
pipeline.py
-----------
The ingestion orchestrator. Wires loaders -> splitter -> embeddings ->
vector_store in the correct order:

    load_pdfs -> split_documents -> (embedding model loaded lazily) -> add_chunks

Run directly for a CLI ingest/reset workflow:
    python -m src.pipeline            # ingest everything in data/raw/
    python -m src.pipeline --reset    # wipe the index, then re-ingest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import settings
from src.loaders import load_pdfs
from src.splitter import split_documents
from src.utils import get_logger
from src.vector_store import add_chunks, collection_count, reset_vector_store

logger = get_logger(__name__)


def run_ingestion_pipeline(source_dir: str | Path | None = None) -> dict:
    """Run the full ingest pipeline end to end and return summary stats."""
    settings.ensure_directories()
    logger.info("Pipeline started")

    documents = load_pdfs(source_dir)
    if not documents:
        logger.warning("No documents were loaded; pipeline stopping early")
        return {
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_added": 0,
            "total_vectors": collection_count(),
        }

    chunks = split_documents(documents)
    added = add_chunks(chunks)
    total = collection_count()

    summary = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "total_vectors": total,
    }
    logger.info("Pipeline finished: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing index before re-ingesting.",
    )
    args = parser.parse_args()

    if args.reset:
        reset_vector_store()
        print("Vector store reset.")

    result = run_ingestion_pipeline()
    print(result)
