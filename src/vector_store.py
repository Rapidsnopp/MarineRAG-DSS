"""Vector store management for MarineRAG-DSS.

Builds and persists a FAISS vector store, and provides a retriever
interface for similarity-based document retrieval.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import config

logger = logging.getLogger(__name__)

_FAISS_INDEX_NAME = "marine_index"


def _get_store_path(store_path: str | None = None) -> Path:
    return Path(store_path or config.VECTOR_STORE_PATH)


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    store_path: str | None = None,
) -> FAISS:
    """Create a FAISS vector store from documents and persist it to disk.

    Args:
        documents: Chunked Document objects to index.
        embeddings: Embeddings model to use for vectorisation.
        store_path: Directory to persist the index.

    Returns:
        The populated FAISS vector store.

    Raises:
        ValueError: If no documents are provided.
    """
    if not documents:
        raise ValueError("Cannot build vector store with an empty document list.")

    logger.info("Building FAISS index from %d chunks…", len(documents))
    vector_store = FAISS.from_documents(documents, embeddings)

    path = _get_store_path(store_path)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path), index_name=_FAISS_INDEX_NAME)
    logger.info("Vector store saved to '%s'.", path)

    return vector_store


def load_vector_store(
    embeddings: Embeddings,
    store_path: str | None = None,
) -> FAISS:
    """Load a previously saved FAISS vector store from disk.

    Args:
        embeddings: Must match the embeddings used when building the store.
        store_path: Directory where the index was saved.

    Returns:
        The loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    path = _get_store_path(store_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{path}'. "
            "Run build_vector_store() first."
        )

    logger.info("Loading FAISS index from '%s'…", path)
    vector_store = FAISS.load_local(
        str(path),
        embeddings,
        index_name=_FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    logger.info("Vector store loaded successfully.")
    return vector_store


def get_or_build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    store_path: str | None = None,
    force_rebuild: bool = False,
) -> FAISS:
    """Return the existing vector store or build one from documents.

    Args:
        documents: Documents used to build the store when needed.
        embeddings: Embeddings model.
        store_path: Persistence directory.
        force_rebuild: If True, always rebuild even if a store exists.

    Returns:
        A ready-to-use FAISS vector store.
    """
    path = _get_store_path(store_path)
    index_file = path / f"{_FAISS_INDEX_NAME}.faiss"

    if not force_rebuild and index_file.exists():
        logger.info("Existing vector store found; loading from disk.")
        return load_vector_store(embeddings, store_path)

    logger.info("No existing vector store found (or rebuild requested); building…")
    return build_vector_store(documents, embeddings, store_path)


def get_retriever(
    vector_store: FAISS,
    k: int | None = None,
) -> VectorStoreRetriever:
    """Return a retriever from the vector store.

    Args:
        vector_store: Populated FAISS vector store.
        k: Number of documents to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever.
    """
    k = k or config.RETRIEVAL_K
    return vector_store.as_retriever(search_kwargs={"k": k})
