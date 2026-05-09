"""API-based LLM module for MarineRAG-DSS.

Uses Google's Gemini API for both language generation and text embeddings.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import config

logger = logging.getLogger(__name__)


def get_api_llm(
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> BaseChatModel:
    """Return a Gemini chat model instance.

    Args:
        model: Model name (e.g. ``"gemini-2.0-flash"``).
               Defaults to ``config.GEMINI_MODEL``.
        temperature: Sampling temperature (0 = deterministic).
        **kwargs: Additional keyword arguments forwarded to ChatGoogleGenerativeAI.

    Returns:
        A configured ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not set.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Provide it via the .env file or as an environment variable."
        )

    model = model or config.GEMINI_MODEL
    logger.info("Initialising Gemini chat model: %s", model)
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=config.GEMINI_API_KEY,
        **kwargs,
    )


def get_api_embeddings(
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """Return a Gemini embeddings instance.

    Args:
        model: Embedding model name.
               Defaults to ``config.GEMINI_EMBEDDING_MODEL``.
        **kwargs: Additional keyword arguments forwarded to GoogleGenerativeAIEmbeddings.

    Returns:
        A configured GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not set.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Provide it via the .env file or as an environment variable."
        )

    model = model or config.GEMINI_EMBEDDING_MODEL
    logger.info("Initialising Gemini embeddings: %s", model)
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=config.GEMINI_API_KEY,
        **kwargs,
    )
