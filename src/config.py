"""
config.py
---------
Single source of truth for every path, model name, and tunable parameter
used across the pipeline. Values can be overridden via environment
variables or a `.env` file (see `.env.example`).
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into the real process environment (os.environ) as early as
# possible. This matters because provider SDKs like `openai` and
# `anthropic` read their API keys directly from os.environ themselves —
# pydantic-settings loading values into *this* Settings object is not
# enough to make them visible there.
load_dotenv()

import os
import certifi

# Some Windows/conda setups leave a stale SSL_CERT_FILE pointing at a
# deleted cert file, which crashes httpx before it even opens a
# connection. If it's missing, replace it with certifi's own bundle.
cert_path = os.environ.get("SSL_CERT_FILE")
if cert_path and not os.path.exists(cert_path):
    os.environ["SSL_CERT_FILE"] = certifi.where()

class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --------------------------------------------------------------- #
    # Paths
    # --------------------------------------------------------------- #
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    results_dir: Path = project_root / "results"
    chroma_persist_dir: Path = project_root / "data" / "processed" / "chroma_db"

    # --------------------------------------------------------------- #
    # Document loading / chunking
    # --------------------------------------------------------------- #
    pdf_glob_pattern: str = "*.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 250

    # --------------------------------------------------------------- #
    # Embeddings
    # --------------------------------------------------------------- #
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True

    # --------------------------------------------------------------- #
    # Vector store
    # --------------------------------------------------------------- #
    collection_name: str = "rag_question_answering"
    distance_metric: str = "cosine"

    # --------------------------------------------------------------- #
    # Retrieval
    # --------------------------------------------------------------- #
    top_k: int = Field(default=4, ge=1, le=50)
    relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # --------------------------------------------------------------- #
    # LLM (answer generation)
    # --------------------------------------------------------------- #
    # Provider string consumed by langchain's `init_chat_model`, e.g.
    # "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5", "ollama:llama3.1".
    llm_model: str = "groq:llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    # Only used when LLM_MODEL starts with "ollama:". Default matches
    # Ollama's own default local server address — override only if
    # Ollama runs elsewhere (a different host, a container, etc.).
    ollama_base_url: str = "http://localhost:11434"

    # --------------------------------------------------------------- #
    # Conversation memory
    # --------------------------------------------------------------- #
    # How many previous (question, answer) turns to keep in context.
    max_history_turns: int = 5
    # Rewrite follow-up questions into standalone questions using the LLM
    # before retrieval (e.g. "what about its risks?" -> "What are the
    # risks of climate change?"). Turn off for a slight speed/cost saving
    # if you don't need multi-turn follow-ups.
    contextualize_followups: bool = True

    # --------------------------------------------------------------- #
    # Refusal message (shown when nothing relevant is retrieved, or the
    # LLM itself determines the context doesn't answer the question)
    # --------------------------------------------------------------- #
    no_answer_message: str = "I could not find this information in the documents."

    # --------------------------------------------------------------- #
    # Logging
    # --------------------------------------------------------------- #
    log_level: str = "INFO"

    def ensure_directories(self) -> None:
        """Create every directory this project writes to, if missing."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.results_dir,
            self.chroma_persist_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
