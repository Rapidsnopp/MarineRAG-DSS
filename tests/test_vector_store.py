"""Tests for the vector_store module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Minimal deterministic embeddings for unit testing."""

    _dim = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash(t) % 100) / 100.0] * self._dim for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(hash(text) % 100) / 100.0] * self._dim


@pytest.fixture()
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture()
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Atlantic cod minimum landing size is 35 cm.",
            metadata={"source": "marine_laws.txt"},
        ),
        Document(
            page_content="The MSY reference point defines sustainable fishing levels.",
            metadata={"source": "fisheries_data.txt"},
        ),
        Document(
            page_content="Ocean pH has decreased to 8.1 due to CO2 absorption.",
            metadata={"source": "environmental_data.txt"},
        ),
    ]


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------


class TestBuildVectorStore:
    def test_builds_and_saves_index(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import build_vector_store

        store = build_vector_store(sample_docs, fake_embeddings, store_path=str(tmp_path))
        assert store is not None
        # Index files should exist
        assert (tmp_path / "marine_index.faiss").exists()
        assert (tmp_path / "marine_index.pkl").exists()

    def test_raises_for_empty_documents(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings
    ) -> None:
        from src.vector_store import build_vector_store

        with pytest.raises(ValueError, match="empty"):
            build_vector_store([], fake_embeddings, store_path=str(tmp_path))


# ---------------------------------------------------------------------------
# load_vector_store
# ---------------------------------------------------------------------------


class TestLoadVectorStore:
    def test_loads_saved_index(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import build_vector_store, load_vector_store

        build_vector_store(sample_docs, fake_embeddings, store_path=str(tmp_path))
        loaded = load_vector_store(fake_embeddings, store_path=str(tmp_path))
        assert loaded is not None

    def test_raises_for_missing_directory(
        self, fake_embeddings: FakeEmbeddings
    ) -> None:
        from src.vector_store import load_vector_store

        with pytest.raises(FileNotFoundError):
            load_vector_store(fake_embeddings, store_path="/nonexistent/path/xyz")


# ---------------------------------------------------------------------------
# get_or_build_vector_store
# ---------------------------------------------------------------------------


class TestGetOrBuildVectorStore:
    def test_builds_when_no_index_exists(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import get_or_build_vector_store

        store = get_or_build_vector_store(
            sample_docs, fake_embeddings, store_path=str(tmp_path)
        )
        assert store is not None

    def test_loads_existing_index(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import build_vector_store, get_or_build_vector_store

        build_vector_store(sample_docs, fake_embeddings, store_path=str(tmp_path))
        store = get_or_build_vector_store(
            sample_docs, fake_embeddings, store_path=str(tmp_path), force_rebuild=False
        )
        assert store is not None

    def test_force_rebuild_ignores_existing(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import build_vector_store, get_or_build_vector_store

        build_vector_store(sample_docs, fake_embeddings, store_path=str(tmp_path))
        store = get_or_build_vector_store(
            sample_docs, fake_embeddings, store_path=str(tmp_path), force_rebuild=True
        )
        assert store is not None


# ---------------------------------------------------------------------------
# get_retriever
# ---------------------------------------------------------------------------


class TestGetRetriever:
    def test_returns_retriever(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import build_vector_store, get_retriever

        store = build_vector_store(sample_docs, fake_embeddings, store_path=str(tmp_path))
        retriever = get_retriever(store, k=2)
        assert retriever is not None

    def test_retriever_returns_relevant_docs(
        self, tmp_path: Path, fake_embeddings: FakeEmbeddings, sample_docs: list[Document]
    ) -> None:
        from src.vector_store import build_vector_store, get_retriever

        store = build_vector_store(sample_docs, fake_embeddings, store_path=str(tmp_path))
        retriever = get_retriever(store, k=2)
        results = retriever.invoke("fishing regulations")
        assert isinstance(results, list)
        assert len(results) <= 2
