"""Document processing utilities for MarineRAG-DSS.

Loads documents from the data directory, splits them into chunks,
and prepares them for embedding and indexing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

from src.config import config

logger = logging.getLogger(__name__)


def load_documents(data_dir: str | None = None) -> list[Document]:
    """Load all supported documents from the given directory.

    Args:
        data_dir: Path to the directory containing documents.
                  Defaults to config.DATA_DIR.

    Returns:
        A list of LangChain Document objects.
    """
    data_dir = data_dir or config.DATA_DIR
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning("Data directory '%s' does not exist.", data_dir)
        return []

    documents: List[Document] = []

    # Load plain text files (.txt, .md)
    for ext in ("*.txt", "*.md"):
        for file_path in data_path.rglob(ext):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata.setdefault("source", str(file_path))
                documents.extend(docs)
                logger.info("Loaded %d chunks from '%s'", len(docs), file_path)
            except Exception:
                logger.exception("Failed to load '%s'", file_path)

    # Load PDF files
    for file_path in data_path.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", str(file_path))
            documents.extend(docs)
            logger.info("Loaded %d pages from PDF '%s'", len(docs), file_path)
        except Exception:
            logger.exception("Failed to load PDF '%s'", file_path)

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target size (in characters) for each chunk.
                    Defaults to config.CHUNK_SIZE.
        chunk_overlap: Number of characters to overlap between chunks.
                       Defaults to config.CHUNK_OVERLAP.

    Returns:
        A list of chunked Document objects.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d documents into %d chunks (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def load_and_split(
    data_dir: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Convenience function: load documents and split into chunks.

    Args:
        data_dir: Directory containing source documents.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    documents = load_documents(data_dir)
    if not documents:
        logger.warning("No documents found; returning empty list.")
        return []
    return split_documents(documents, chunk_size, chunk_overlap)
