"""Tests for the document_processor module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.document_processor import load_and_split, load_documents, split_documents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_doc_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample text files."""
    (tmp_path / "doc1.txt").write_text(
        "Marine protected areas are zones where human activities are restricted. "
        "They help conserve biodiversity and support sustainable fisheries.",
        encoding="utf-8",
    )
    (tmp_path / "doc2.txt").write_text(
        "The UNCLOS convention governs ocean resource management. "
        "Article 61 describes the conservation of living resources in the EEZ.",
        encoding="utf-8",
    )
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "doc3.txt").write_text(
        "Atlantic cod is a commercially important demersal fish species.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def sample_documents() -> list[Document]:
    """Return a list of sample Document objects."""
    return [
        Document(
            page_content="Ocean temperatures have increased by 1.2°C over the last century.",
            metadata={"source": "environmental_data.txt"},
        ),
        Document(
            page_content=(
                "The Common Fisheries Policy sets maximum sustainable yield targets. "
                "Fishing vessels must hold a valid licence. "
                "Electronic logbooks are mandatory for vessels over 10 m."
            ),
            metadata={"source": "marine_laws.txt"},
        ),
    ]


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_loads_txt_files(self, temp_doc_dir: Path) -> None:
        docs = load_documents(str(temp_doc_dir))
        assert len(docs) >= 3

    def test_returns_empty_for_missing_dir(self) -> None:
        docs = load_documents("/nonexistent/path/xyz")
        assert docs == []

    def test_documents_have_source_metadata(self, temp_doc_dir: Path) -> None:
        docs = load_documents(str(temp_doc_dir))
        for doc in docs:
            assert "source" in doc.metadata

    def test_loads_files_in_subdirectory(self, temp_doc_dir: Path) -> None:
        docs = load_documents(str(temp_doc_dir))
        sources = [doc.metadata["source"] for doc in docs]
        assert any("subdir" in s for s in sources)


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    def test_splits_into_chunks(self, sample_documents: list[Document]) -> None:
        chunks = split_documents(sample_documents, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= len(sample_documents)

    def test_chunks_respect_size(self, sample_documents: list[Document]) -> None:
        chunk_size = 100
        chunks = split_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 2  # allow some tolerance

    def test_returns_empty_for_empty_input(self) -> None:
        chunks = split_documents([], chunk_size=500)
        assert chunks == []

    def test_preserves_metadata(self, sample_documents: list[Document]) -> None:
        chunks = split_documents(sample_documents, chunk_size=50, chunk_overlap=5)
        sources = {chunk.metadata.get("source") for chunk in chunks}
        assert "environmental_data.txt" in sources or "marine_laws.txt" in sources


# ---------------------------------------------------------------------------
# load_and_split (integration)
# ---------------------------------------------------------------------------


class TestLoadAndSplit:
    def test_returns_chunks_from_directory(self, temp_doc_dir: Path) -> None:
        chunks = load_and_split(str(temp_doc_dir), chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 0

    def test_returns_empty_for_missing_dir(self) -> None:
        chunks = load_and_split("/nonexistent/path/xyz")
        assert chunks == []
