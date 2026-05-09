"""Tests for the kg_store module."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document


@pytest.fixture()
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "Atlantic Cod is managed under the Common Fisheries Policy. "
                "The minimum conservation reference size is 35 cm."
            ),
            metadata={"source": "marine_laws.txt", "chunk_id": "marine_laws.txt::chunk-0000", "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Bluefin tuna stocks are monitored by ICCAT and the International Whaling Commission. "
                "MARPOL restricts plastic discharge."
            ),
            metadata={"source": "fisheries_data.txt", "chunk_id": "fisheries_data.txt::chunk-0001", "chunk_index": 1},
        ),
        Document(
            page_content=(
                "Marine protected areas support biodiversity, reduce pollution, and help ecosystem resilience."
            ),
            metadata={"source": "environmental_data.txt", "chunk_id": "environmental_data.txt::chunk-0002", "chunk_index": 2},
        ),
    ]


class TestKnowledgeGraphStore:
    def test_extract_entities_finds_expected_terms(self) -> None:
        from src.kg_store import extract_entities

        entities = extract_entities("Atlantic Cod is managed under the Common Fisheries Policy and MARPOL.")
        assert "Atlantic Cod" in entities
        assert "Common Fisheries Policy" in entities
        assert "MARPOL" in entities

    def test_build_save_and_load(self, tmp_path: Path, sample_docs: list[Document]) -> None:
        from src.kg_store import KnowledgeGraphStore

        graph = KnowledgeGraphStore.build(sample_docs)
        assert graph.chunk_to_entities
        graph.save(tmp_path)

        loaded = KnowledgeGraphStore.load(tmp_path)
        assert loaded.chunk_to_entities == graph.chunk_to_entities

    def test_expand_documents_adds_related_chunks(self, sample_docs: list[Document]) -> None:
        from src.kg_store import KnowledgeGraphStore

        graph = KnowledgeGraphStore.build(sample_docs)
        seed_docs = [sample_docs[0]]
        expanded = graph.expand_documents(seed_docs, question="What about tuna and MARPOL rules?", max_related_chunks=2)
        assert len(expanded) >= 1
        assert expanded[0].metadata["chunk_id"] == "marine_laws.txt::chunk-0000"

    def test_describe_context_returns_summary(self, sample_docs: list[Document]) -> None:
        from src.kg_store import KnowledgeGraphStore

        graph = KnowledgeGraphStore.build(sample_docs)
        summary = graph.describe_context("Tell me about Atlantic Cod", sample_docs[:2])
        assert "Question entities" in summary
        assert "Graph entities" in summary