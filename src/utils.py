"""
utils.py
--------
Small, dependency-light helpers shared across the project: logging
setup, text cleaning, and truncation for display. Keep this generic —
it should not know about PDFs, embeddings, Chroma, or the LLM.
"""

from __future__ import annotations

import logging
import re
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, shared format across modules."""
    from src.config import settings

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level.upper())
        logger.propagate = False
    return logger


def clean_text(text: str) -> str:
    """Normalize whitespace and strip odd PDF extraction artifacts."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def truncate(text: str, max_chars: int = 400) -> str:
    """Shorten text for display, adding an ellipsis if truncated."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
