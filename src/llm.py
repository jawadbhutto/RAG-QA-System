"""
llm.py
------
Single responsibility: provide the chat model used to generate answers.

Uses LangChain's `init_chat_model`, which accepts a "provider:model"
string and picks the right integration package underneath. This keeps
the rest of the codebase provider-agnostic — swapping OpenAI for
Anthropic or a local Ollama model is a config change, not a code change.

Requires the matching provider package to be installed, e.g.:
    langchain-openai     for "openai:..."
    langchain-anthropic   for "anthropic:..."
    langchain-ollama       for "ollama:..." (local models, no API key)

Ollama tag names (e.g. "qwen2.5:3b") already contain a colon, so we
split the provider prefix off manually and pass `model_provider`
explicitly rather than relying on init_chat_model to guess it from a
string with two colons in it. We also translate `llm_max_tokens` into
whichever kwarg the target provider actually expects — OpenAI/Anthropic
use `max_tokens`, Ollama uses `num_predict`.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


def _parse_model_string(model_string: str) -> tuple[str, str]:
    """Split "provider:model" into (provider, model), splitting only once."""
    provider, _, model_name = model_string.partition(":")
    if not model_name:
        raise ValueError(
            f"LLM_MODEL must be in 'provider:model' form, got: {model_string!r}"
        )
    return provider, model_name


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached chat model configured from `settings.llm_model`."""
    from langchain.chat_models import init_chat_model

    provider, model_name = _parse_model_string(settings.llm_model)

    kwargs: dict = {"temperature": settings.llm_temperature}
    if provider == "ollama":
        # ChatOllama has no `max_tokens` kwarg — the equivalent is `num_predict`.
        kwargs["num_predict"] = settings.llm_max_tokens
        if settings.ollama_base_url:
            kwargs["base_url"] = settings.ollama_base_url
    else:
        kwargs["max_tokens"] = settings.llm_max_tokens

    logger.info(
        "Loading LLM provider=%s model=%s (temperature=%s, kwargs=%s)",
        provider,
        model_name,
        settings.llm_temperature,
        {k: v for k, v in kwargs.items() if k != "temperature"},
    )
    return init_chat_model(model_name, model_provider=provider, **kwargs)
