"""Local LLM module for MarineRAG-DSS.

Uses Ollama to run LLMs and produce embeddings entirely on the local
machine — no external API calls required.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import config

logger = logging.getLogger(__name__)


def get_local_llm(
    model: str | None = None,
    temperature: float = 0.2,
    base_url: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Return an Ollama chat model instance.

    Args:
        model: Ollama model tag (e.g. ``"llama3.2"``).
               Defaults to ``config.OLLAMA_MODEL``.
        temperature: Sampling temperature (0 = deterministic).
        base_url: Ollama server URL.
                  Defaults to ``config.OLLAMA_BASE_URL``.
        **kwargs: Additional keyword arguments forwarded to ChatOllama.

    Returns:
        A configured ChatOllama instance.
    """
    model = model or config.OLLAMA_MODEL
    base_url = base_url or config.OLLAMA_BASE_URL
    logger.info("Initialising Ollama chat model: %s @ %s", model, base_url)
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        **kwargs,
    )


def get_local_embeddings(
    model: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> Embeddings:
    """Return an Ollama embeddings instance.

    Args:
        model: Ollama embedding model tag (e.g. ``"nomic-embed-text"``).
               Defaults to ``config.OLLAMA_EMBEDDING_MODEL``.
        base_url: Ollama server URL.
                  Defaults to ``config.OLLAMA_BASE_URL``.
        **kwargs: Additional keyword arguments forwarded to OllamaEmbeddings.

    Returns:
        A configured OllamaEmbeddings instance.
    """
    model = model or config.OLLAMA_EMBEDDING_MODEL
    base_url = base_url or config.OLLAMA_BASE_URL
    logger.info("Initialising Ollama embeddings: %s @ %s", model, base_url)
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
        **kwargs,
    )
