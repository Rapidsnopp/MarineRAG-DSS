"""
Handle embeddings creation and storage for RAG documents.

Supports multiple embedding providers:
- Gemini API (via langchain-google-genai)
- Ollama (local)
- HuggingFace sentence-transformers
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingsHandler:
    """Handle embeddings creation for documents."""

    def __init__(self, embeddings_provider: Embeddings | None = None):
        """Initialize embeddings handler.

        Args:
            embeddings_provider: Embeddings provider instance.
                If None, will be auto-detected based on config.
        """
        self.embeddings_provider = embeddings_provider
        self.embeddings_cache: dict[str, list[float]] = {}
        self.embeddings_metadata: dict[str, dict[str, Any]] = {}

    def set_embeddings_provider(self, provider: Embeddings) -> None:
        """Set the embeddings provider.

        Args:
            provider: Embeddings provider instance
        """
        self.embeddings_provider = provider

    def embed_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Create embeddings for documents.

        Args:
            documents: List of documents to embed

        Returns:
            Dictionary with embeddings data
        """
        if not self.embeddings_provider:
            logger.error("Embeddings provider not set")
            return {"success": False, "error": "Embeddings provider not set"}

        try:
            # Extract document contents
            contents = [doc.page_content for doc in documents]

            logger.info(f"Creating embeddings for {len(documents)} documents...")

            # Try batch embedding first
            embeddings_list = []
            try:
                embeddings_list = self.embeddings_provider.embed_documents(contents)
                if len(embeddings_list) == 1 and len(contents) > 1:
                    logger.warning(
                        "embeddings API returned only 1 embedding for multiple documents. "
                        "Falling back to individual embedding..."
                    )
                    embeddings_list = []
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}. Trying individual embedding...")

            # Fall back to individual embedding if batch didn't work
            if not embeddings_list or len(embeddings_list) != len(contents):
                logger.info("Embedding documents individually...")
                embeddings_list = []
                for i, content in enumerate(contents):
                    try:
                        embedding = self.embeddings_provider.embed_query(content)
                        embeddings_list.append(embedding)
                        if (i + 1) % 10 == 0:
                            logger.debug(f"Embedded {i + 1}/{len(contents)} documents")
                    except Exception as e:
                        logger.error(f"Failed to embed document {i}: {e}")
                        # Use zero vector as fallback
                        embeddings_list.append([0.0] * 3072)

            # Ensure we have the same number of embeddings as documents
            if len(embeddings_list) != len(documents):
                logger.warning(
                    f"Mismatch: {len(documents)} docs but {len(embeddings_list)} embeddings"
                )

            # Store embeddings with metadata
            count = 0
            for doc, embedding in zip(documents, embeddings_list):
                doc_id = self._get_doc_id(doc)
                self.embeddings_cache[doc_id] = embedding
                self.embeddings_metadata[doc_id] = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding_dim": len(embedding),
                }
                count += 1

            logger.info(
                f"Successfully created {count} embeddings "
                f"(dimension: {len(embeddings_list[0]) if embeddings_list else 0})"
            )
            logger.info(f"Total in cache: {len(self.embeddings_cache)} embeddings")

            return {
                "success": True,
                "total": count,
                "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
            }

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _get_doc_id(self, doc: Document) -> str:
        """Generate unique ID for document.

        Args:
            doc: Document

        Returns:
            Unique document ID
        """
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        param = metadata.get("parameter", "").replace(" ", "_").lower()
        species = metadata.get("species_code", "unknown")
        unit = metadata.get("unit", "").replace(" ", "_").replace("/", "_").lower()

        # Create a unique hash from all components
        doc_id = f"{source}_{species}_{param}_{unit}"
        # Ensure it's safe for use as a filename
        doc_id = doc_id.replace(" ", "_").replace("/", "_")

        logger.debug(f"Generated doc_id: {doc_id}")
        return doc_id

    def save_embeddings(
        self,
        output_dir: str = "data/embeddings",
    ) -> Path:
        """Save embeddings to disk.

        Args:
            output_dir: Directory to save embeddings

        Returns:
            Path to saved embeddings file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_file = output_path / "growth_thresholds_embeddings.json"
        with open(embeddings_file, "w", encoding="utf-8") as f:
            json.dump(self.embeddings_cache, f, ensure_ascii=False)

        # Save metadata
        metadata_file = output_path / "growth_thresholds_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.embeddings_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self.embeddings_cache)} embeddings to {embeddings_file}")
        logger.info(f"Saved metadata to {metadata_file}")

        return embeddings_file

    def load_embeddings(self, embeddings_file: str | Path) -> bool:
        """Load embeddings from disk.

        Args:
            embeddings_file: Path to embeddings file

        Returns:
            True if successful
        """
        try:
            embeddings_file = Path(embeddings_file)
            with open(embeddings_file, "r", encoding="utf-8") as f:
                self.embeddings_cache = json.load(f)

            metadata_file = embeddings_file.parent / "growth_thresholds_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.embeddings_metadata = json.load(f)

            logger.info(f"Loaded {len(self.embeddings_cache)} embeddings")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get statistics about embeddings.

        Returns:
            Statistics dictionary
        """
        if not self.embeddings_cache:
            return {"total": 0, "embedding_dimension": 0}

        first_embedding = next(iter(self.embeddings_cache.values()), [])
        return {
            "total": len(self.embeddings_cache),
            "embedding_dimension": len(first_embedding) if first_embedding else 0,
            "storage_size_kb": sum(
                len(json.dumps(v)) for v in self.embeddings_cache.values()
            ) / 1024,
        }


def create_embeddings_from_config() -> Embeddings | None:
    """Create embeddings provider from environment config.

    Returns:
        Embeddings provider or None
    """
    from src.config import config

    try:
        # Try Gemini API first
        if config.GEMINI_API_KEY:
            logger.info(f"Using Gemini embeddings: {config.GEMINI_EMBEDDING_MODEL}")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=config.GEMINI_EMBEDDING_MODEL,
                google_api_key=config.GEMINI_API_KEY,
            )
    except Exception as e:
        logger.warning(f"Gemini embeddings not available: {e}")

    try:
        # Try Ollama embeddings
        embeddings_provider = (config.LOCAL_EMBEDDINGS_PROVIDER or "ollama").strip().lower()
        if embeddings_provider in {"hf", "huggingface", "sentence-transformers", "sbert"}:
            logger.info(f"Using HuggingFace embeddings: {config.HF_EMBEDDING_MODEL}")
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
        else:
            logger.info(f"Using Ollama embeddings: {config.OLLAMA_EMBEDDING_MODEL}")
            from langchain_community.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            )
    except Exception as e:
        logger.warning(f"Local embeddings not available: {e}")

    logger.error("No embeddings provider available!")
    return None
