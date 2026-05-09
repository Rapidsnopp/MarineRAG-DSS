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
    """Return a local embeddings instance.

    Args:
        model: Embedding model name.
               Defaults to ``config.OLLAMA_EMBEDDING_MODEL`` or
               ``config.HF_EMBEDDING_MODEL`` based on provider.
        base_url: Ollama server URL (only used for the Ollama provider).
                  Defaults to ``config.OLLAMA_BASE_URL``.
        **kwargs: Additional keyword arguments forwarded to the embeddings provider.

    Returns:
        A configured embeddings instance (Ollama or HuggingFace).
    """
    provider = (config.LOCAL_EMBEDDINGS_PROVIDER or "ollama").strip().lower()
    if provider in {"hf", "huggingface", "sentence-transformers", "sbert"}:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model = model or config.HF_EMBEDDING_MODEL
        model_kwargs = {
            "device": config.HF_EMBEDDING_DEVICE,
            "trust_remote_code": config.HF_EMBEDDING_TRUST_REMOTE_CODE,
        }
        model_kwargs.update(kwargs.pop("model_kwargs", {}))
        logger.info("Initialising HuggingFace embeddings: %s", model)
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    model = model or config.OLLAMA_EMBEDDING_MODEL
    base_url = base_url or config.OLLAMA_BASE_URL
    logger.info("Initialising Ollama embeddings: %s @ %s", model, base_url)
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
        **kwargs,
    )
