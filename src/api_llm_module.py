"""API-based LLM module for MarineRAG-DSS.

Uses OpenAI's API (or any OpenAI-compatible endpoint) for both
language generation and text embeddings.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import config

logger = logging.getLogger(__name__)


def get_api_llm(
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> BaseChatModel:
    """Return an OpenAI chat model instance.

    Args:
        model: Model name (e.g. ``"gpt-4o-mini"``).
               Defaults to ``config.OPENAI_MODEL``.
        temperature: Sampling temperature (0 = deterministic).
        **kwargs: Additional keyword arguments forwarded to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Provide it via the .env file or as an environment variable."
        )

    model = model or config.OPENAI_MODEL
    logger.info("Initialising OpenAI chat model: %s", model)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
        **kwargs,
    )


def get_api_embeddings(
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """Return an OpenAI embeddings instance.

    Args:
        model: Embedding model name.
               Defaults to ``config.OPENAI_EMBEDDING_MODEL``.
        **kwargs: Additional keyword arguments forwarded to OpenAIEmbeddings.

    Returns:
        A configured OpenAIEmbeddings instance.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Provide it via the .env file or as an environment variable."
        )

    model = model or config.OPENAI_EMBEDDING_MODEL
    logger.info("Initialising OpenAI embeddings: %s", model)
    return OpenAIEmbeddings(
        model=model,
        api_key=config.OPENAI_API_KEY,
        **kwargs,
    )
