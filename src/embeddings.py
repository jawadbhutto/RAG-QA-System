"""
embeddings.py
-------------
Single responsibility: provide the embedding model used to turn text
into vectors. Cached so the (relatively expensive) model load happens
once per process.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return a cached `HuggingFaceEmbeddings` instance."""
    logger.info(
        "Loading embedding model '%s' on device '%s'",
        settings.embedding_model_name,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": settings.normalize_embeddings},
    )
