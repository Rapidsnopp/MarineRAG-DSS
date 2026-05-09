"""Vector store management for MarineRAG-DSS.

Builds and persists a FAISS similarity index, and provides a retriever
interface for semantic document retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

from src.config import config

logger = logging.getLogger(__name__)

_FAISS_INDEX_NAME = "marine_index"


def _get_store_path(store_path: str | None = None) -> Path:
    return Path(store_path or config.VECTOR_STORE_PATH)


def _index_file(path: Path, index_name: str = _FAISS_INDEX_NAME) -> Path:
    return path / f"{index_name}.faiss"


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    store_path: str | None = None,
) -> FAISS:
    """Create a FAISS vector store from documents and persist it to disk."""
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
    """Load a previously saved FAISS vector store from disk."""
    path = _get_store_path(store_path)
    index_file = _index_file(path)
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{path}'. "
            "Run build_vector_store() first."
        )

    logger.info("Loading vector index from '%s'…", path)
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
    """Return the existing vector store or build one from documents."""
    path = _get_store_path(store_path)
    index_file = _index_file(path)

    if not force_rebuild and index_file.exists():
        logger.info("Existing vector store found; loading from disk.")
        try:
            return load_vector_store(embeddings, store_path)
        except Exception as exc:  # pragma: no cover - safety fallback
            logger.warning("Failed to load vector store at '%s': %s", path, exc)
            logger.info("Rebuilding vector store due to load failure.")

    logger.info("No existing vector store found (or rebuild requested); building…")
    return build_vector_store(documents, embeddings, store_path)


def get_retriever(
    vector_store: FAISS,
    k: int | None = None,
) -> VectorStoreRetriever:
    """Return a retriever from the vector store."""
    k = k or config.RETRIEVAL_K
    return vector_store.as_retriever(search_kwargs={"k": k})
