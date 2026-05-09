"""Knowledge graph utilities for MarineRAG-DSS.

This module builds a lightweight knowledge graph from retrieved chunks,
then uses graph connectivity to expand the semantic retrieval set.
The implementation stays dependency-light so it can run in the existing
project without introducing a large graph framework.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)

_ENTITY_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "By",
    "For",
    "From",
    "In",
    "Into",
    "Of",
    "On",
    "Or",
    "The",
    "To",
    "With",
    "Within",
    "Cac",
    "Của",
    "Các",
    "Do",
    "Den",
    "Đến",
    "Mot",
    "Một",
    "Nhung",
    "Những",
    "Tai",
    "Tại",
    "Theo",
    "Trong",
    "Tu",
    "Từ",
    "Va",
    "Và",
    "Ve",
    "Về",
    "Voi",
    "Với",
}

_CONNECTORS = {
    "and",
    "of",
    "the",
    "for",
    "to",
    "in",
    "on",
    "with",
    "by",
    "from",
    "de",
    "la",
    "van",
    "von",
    "et",
    "di",
    "du",
    "a",
    "an",
    "va",
    "và",
    "cua",
    "của",
    "trong",
    "tai",
    "tại",
    "cho",
    "den",
    "đến",
    "tu",
    "từ",
    "voi",
    "với",
    "tren",
    "trên",
    "duoi",
    "dưới",
    "theo",
    "ve",
    "về",
    "do",
    "cac",
    "các",
    "nhung",
    "những",
    "mot",
    "một",
    "nam",
    "năm",
    "khi",
}

_MARINE_TERMS = {
    "anchovy",
    "beaked",
    "bluefin",
    "boarfish",
    "bream",
    "butterfish",
    "cartilaginous",
    "clam",
    "cod",
    "crab",
    "dolphin",
    "eel",
    "flatfish",
    "grouper",
    "hake",
    "herring",
    "lobster",
    "mackerel",
    "marine",
    "mollusc",
    "mollusk",
    "octopus",
    "pollock",
    "ray",
    "salmon",
    "sardine",
    "sea",
    "seabass",
    "shark",
    "shrimp",
    "squid",
    "sturgeon",
    "tuna",
    "whale",
}

_VI_SHORT_TERMS = {
    "cá",
    "ốc",
    "sò",
}

_VI_MARINE_TERMS = {
    "cá",
    "tôm",
    "ngao",
    "hàu",
    "mực",
    "ốc",
    "sò",
    "cua",
    "ghẹ",
    "rong",
    "tảo",
}

_VI_DOMAIN_PHRASES = {
    "thủy sản",
    "hải sản",
    "nhuyễn thể",
    "nuôi trồng",
    "nuôi trồng thủy sản",
    "ao nuôi",
    "lồng bè",
    "nuôi cá",
    "nuôi cá lồng",
    "nuôi tôm",
    "nuôi ngao",
    "nuôi hàu",
    "chất lượng nước",
    "môi trường",
    "quan trắc",
    "rừng ngập mặn",
    "bãi triều",
    "nước lợ",
    "nước mặn",
    "nước ngọt",
}


def _normalise_entity(entity: str) -> str:
    entity = re.sub(r"\s+", " ", entity.strip(" \t\r\n,.;:()[]{}<>"))
    entity = re.sub(
        r"^(?:the|a|an|mot|một|cac|các|nhung|những)\s+",
        "",
        entity,
        flags=re.IGNORECASE,
    )
    return entity.strip()


def extract_entities(text: str) -> list[str]:
    """Extract lightweight entity candidates from text."""
    candidates: list[str] = []

    for acronym in re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text):
        if len(acronym) >= 2:
            candidates.append(acronym)

    phrase_pattern = re.compile(
        r"\b(?:[A-Z][\w&/-]*|[A-Z]{2,})(?:\s+(?:[A-Z][\w&/-]*|[A-Z]{2,}))+"
    )
    for match in phrase_pattern.findall(text):
        phrase = _normalise_entity(match)
        if phrase and phrase not in _ENTITY_STOPWORDS:
            candidates.append(phrase)

    for match in re.findall(
        r"\b(?:QCVN|TCVN)\s*[\d:./-]+(?:/[A-Z0-9-]+)?\b",
        text,
        flags=re.IGNORECASE,
    ):
        phrase = _normalise_entity(match)
        if phrase:
            candidates.append(phrase)

    species_pattern = re.compile(
        r"\b([A-Z][a-z]+\s+(?:" + "|".join(sorted(_MARINE_TERMS)) + r"))\b",
        flags=re.IGNORECASE,
    )
    for match in species_pattern.findall(text):
        phrase = _normalise_entity(match)
        if phrase:
            candidates.append(phrase)

    vi_term_pattern = re.compile(
        r"\b(?:" + "|".join(sorted(_VI_MARINE_TERMS)) + r")\b(?:\s+[\wÀ-ỹ-]+){0,3}",
        flags=re.IGNORECASE,
    )
    for match in vi_term_pattern.findall(text):
        phrase = _normalise_entity(match)
        if phrase:
            candidates.append(phrase)

    lower_text = text.lower()
    for phrase in _VI_DOMAIN_PHRASES:
        if phrase in lower_text:
            candidates.append(phrase)

    deduped: list[str] = []
    seen = set()
    connector_set = {item.lower() for item in _CONNECTORS}
    for candidate in candidates:
        if not candidate:
            continue
        candidate = _normalise_entity(candidate)
        if not candidate or candidate.lower() in connector_set:
            continue
        if len(candidate) < 3 and candidate.lower() not in _VI_SHORT_TERMS:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    return deduped


def _ensure_document_metadata(document: Document, chunk_id: str, chunk_index: int) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    metadata.setdefault("chunk_id", chunk_id)
    metadata.setdefault("chunk_index", chunk_index)
    return metadata


@dataclass
class KnowledgeGraphStore:
    """Serialized lightweight graph index over document chunks."""

    entity_to_chunks: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    chunk_to_entities: dict[str, list[str]] = field(default_factory=dict)
    chunk_text: dict[str, str] = field(default_factory=dict)
    chunk_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    entity_neighbors: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))

    @classmethod
    def build(cls, documents: list[Document]) -> "KnowledgeGraphStore":
        store = cls()

        for index, document in enumerate(documents):
            metadata = dict(document.metadata or {})
            source = metadata.get("source", "unknown_source")
            chunk_id = metadata.get("chunk_id") or f"{source}::chunk-{index:04d}"

            entities = extract_entities(document.page_content)
            store.chunk_to_entities[chunk_id] = entities
            store.chunk_text[chunk_id] = document.page_content
            store.chunk_metadata[chunk_id] = _ensure_document_metadata(document, chunk_id, index)

            for entity in entities:
                store.entity_to_chunks[entity].add(chunk_id)

            for left, right in combinations(entities, 2):
                store.entity_neighbors[left][right] = store.entity_neighbors[left].get(right, 0) + 1
                store.entity_neighbors[right][left] = store.entity_neighbors[right].get(left, 0) + 1

        logger.info(
            "Built knowledge graph with %d entities and %d chunks.",
            len(store.entity_to_chunks),
            len(store.chunk_to_entities),
        )
        return store

    @classmethod
    def load(cls, store_path: str | Path) -> "KnowledgeGraphStore":
        path = Path(store_path)
        graph_file = path / "knowledge_graph.json"
        if not graph_file.exists():
            raise FileNotFoundError(f"Knowledge graph not found at '{graph_file}'.")

        payload = json.loads(graph_file.read_text(encoding="utf-8"))
        store = cls()
        store.entity_to_chunks = defaultdict(
            set,
            {entity: set(chunk_ids) for entity, chunk_ids in payload.get("entity_to_chunks", {}).items()},
        )
        store.chunk_to_entities = {
            chunk_id: list(entities) for chunk_id, entities in payload.get("chunk_to_entities", {}).items()
        }
        store.chunk_text = payload.get("chunk_text", {})
        store.chunk_metadata = payload.get("chunk_metadata", {})
        store.entity_neighbors = defaultdict(
            dict,
            {
                entity: {neighbor: int(weight) for neighbor, weight in neighbors.items()}
                for entity, neighbors in payload.get("entity_neighbors", {}).items()
            },
        )
        logger.info("Knowledge graph loaded from '%s'.", graph_file)
        return store

    def save(self, store_path: str | Path) -> None:
        path = Path(store_path)
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "entity_to_chunks": {
                entity: sorted(chunk_ids) for entity, chunk_ids in self.entity_to_chunks.items()
            },
            "chunk_to_entities": self.chunk_to_entities,
            "chunk_text": self.chunk_text,
            "chunk_metadata": self.chunk_metadata,
            "entity_neighbors": self.entity_neighbors,
        }

        graph_file = path / "knowledge_graph.json"
        graph_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Knowledge graph saved to '%s'.", graph_file)

    def _question_entities(self, question: str | None) -> list[str]:
        if not question:
            return []
        return extract_entities(question)

    def _rank_candidate_chunks(
        self,
        seed_entities: list[str],
        question_entities: list[str],
        selected_chunk_ids: set[str],
    ) -> list[tuple[int, int, str]]:
        candidate_scores: Counter[str] = Counter()
        focus_entities = list(dict.fromkeys(seed_entities + question_entities))

        for entity in focus_entities:
            neighbors = self.entity_neighbors.get(entity, {})
            for neighbor, weight in neighbors.items():
                candidate_scores[neighbor] += weight

        if not candidate_scores:
            for entity in focus_entities:
                candidate_scores[entity] += 1

        ranked_chunks: list[tuple[int, int, str]] = []
        for entity, entity_score in candidate_scores.most_common():
            for chunk_id in self.entity_to_chunks.get(entity, set()):
                if chunk_id in selected_chunk_ids:
                    continue
                chunk_entities = self.chunk_to_entities.get(chunk_id, [])
                overlap = len(set(chunk_entities) & set(focus_entities))
                score = entity_score + overlap
                chunk_index = int(self.chunk_metadata.get(chunk_id, {}).get("chunk_index", 0))
                ranked_chunks.append((score, chunk_index, chunk_id))

        ranked_chunks.sort(key=lambda item: (-item[0], item[1], item[2]))
        return ranked_chunks

    def expand_documents(
        self,
        seed_documents: list[Document],
        question: str | None = None,
        max_related_chunks: int | None = None,
    ) -> list[Document]:
        if not seed_documents:
            return []

        max_related_chunks = max_related_chunks or config.KG_MAX_EXPANSION_CHUNKS
        selected: dict[str, Document] = {}

        for document in seed_documents:
            metadata = dict(document.metadata or {})
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                source = metadata.get("source", "unknown_source")
                chunk_id = f"{source}::chunk-{len(selected):04d}"
                metadata["chunk_id"] = chunk_id
            selected.setdefault(chunk_id, Document(page_content=document.page_content, metadata=metadata))

        seed_entities: list[str] = []
        for document in seed_documents:
            metadata = document.metadata or {}
            chunk_id = metadata.get("chunk_id")
            if chunk_id and chunk_id in self.chunk_to_entities:
                seed_entities.extend(self.chunk_to_entities.get(chunk_id, []))
            else:
                seed_entities.extend(extract_entities(document.page_content))

        question_entities = self._question_entities(question)
        ranked_chunks = self._rank_candidate_chunks(seed_entities, question_entities, set(selected))

        for _, _, chunk_id in ranked_chunks[:max_related_chunks]:
            metadata = dict(self.chunk_metadata.get(chunk_id, {}))
            if not metadata:
                metadata = {"chunk_id": chunk_id}
            selected.setdefault(
                chunk_id,
                Document(page_content=self.chunk_text.get(chunk_id, ""), metadata=metadata),
            )

        ordered_documents = sorted(
            selected.values(),
            key=lambda document: (
                int(document.metadata.get("chunk_index", 0)),
                str(document.metadata.get("chunk_id", "")),
            ),
        )
        return ordered_documents

    def describe_context(
        self,
        question: str | None,
        documents: list[Document],
        max_entities: int | None = None,
        max_relations: int | None = None,
    ) -> str:
        if not documents:
            return "No graph context available."

        max_entities = max_entities or config.KG_MAX_GRAPH_ENTITIES
        max_relations = max_relations or config.KG_MAX_EXPANSION_CHUNKS

        entity_counts: Counter[str] = Counter()
        for document in documents:
            chunk_id = (document.metadata or {}).get("chunk_id")
            if chunk_id:
                entity_counts.update(self.chunk_to_entities.get(chunk_id, extract_entities(document.page_content)))

        question_entities = self._question_entities(question)
        for entity in question_entities:
            entity_counts[entity] += 2

        top_entities = [entity for entity, _ in entity_counts.most_common(max_entities)]
        relation_lines: list[str] = []

        for left, right in combinations(top_entities, 2):
            weight = self.entity_neighbors.get(left, {}).get(right)
            if weight:
                relation_lines.append(f"{left} <-> {right} (shared chunks: {weight})")
            if len(relation_lines) >= max_relations:
                break

        lines = [
            f"Question entities: {', '.join(question_entities) if question_entities else 'None detected'}",
            f"Graph entities: {', '.join(top_entities) if top_entities else 'None detected'}",
        ]
        if relation_lines:
            lines.append("Graph relations:")
            lines.extend(f"- {line}" for line in relation_lines)
        return "\n".join(lines)


class Neo4jKnowledgeGraphStore:
    """Neo4j-backed knowledge graph store."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        batch_size: int = 500,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.batch_size = batch_size
        self._schema_ready = False
        self._driver = self._create_driver()

    @classmethod
    def from_config(cls) -> "Neo4jKnowledgeGraphStore":
        if not config.NEO4J_URI:
            raise ValueError("NEO4J_URI is not set.")
        if not config.NEO4J_USER:
            raise ValueError("NEO4J_USER is not set.")
        return cls(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE,
            batch_size=config.NEO4J_BATCH_SIZE,
        )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def needs_ingest(self, force_rebuild: bool) -> bool:
        if force_rebuild:
            return True
        return self.is_empty()

    def is_empty(self) -> bool:
        rows = self._run_read("MATCH (n) RETURN count(n) AS count")
        if not rows:
            return True
        return int(rows[0].get("count", 0)) == 0

    def chunk_count(self) -> int:
        rows = self._run_read("MATCH (c:Chunk) RETURN count(c) AS count")
        if not rows:
            return 0
        return int(rows[0].get("count", 0))

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self._run_write(
            "CREATE CONSTRAINT kg_entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        self._run_write(
            "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        self._schema_ready = True

    def clear(self) -> None:
        self._run_write("MATCH (n) DETACH DELETE n")

    def build(
        self,
        documents: list[Document],
        clear: bool = False,
    ) -> "Neo4jKnowledgeGraphStore":
        if not documents:
            raise ValueError("Cannot build knowledge graph from an empty document list.")

        graph = KnowledgeGraphStore.build(documents)
        self.ingest_graph(graph, clear=clear)
        return self

    def ingest_graph(self, graph: KnowledgeGraphStore, clear: bool = False) -> None:
        self.ensure_schema()
        if clear:
            self.clear()

        rows: list[dict[str, Any]] = []
        for chunk_id, entities in graph.chunk_to_entities.items():
            metadata = dict(graph.chunk_metadata.get(chunk_id, {}))
            metadata.setdefault("chunk_id", chunk_id)
            if "chunk_index" not in metadata:
                metadata["chunk_index"] = self._safe_int(metadata.get("chunk_index", 0))

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": graph.chunk_text.get(chunk_id, ""),
                    "source": metadata.get("source"),
                    "chunk_index": self._safe_int(metadata.get("chunk_index", 0)),
                    "metadata": self._encode_metadata(metadata),
                    "entities": list(entities),
                }
            )

            if len(rows) >= self.batch_size:
                self._upsert_chunks(rows)
                rows = []

        if rows:
            self._upsert_chunks(rows)

        edges: list[dict[str, Any]] = []
        for left, neighbors in graph.entity_neighbors.items():
            for right, weight in neighbors.items():
                if left >= right:
                    continue
                edges.append({"left": left, "right": right, "weight": int(weight)})
                if len(edges) >= self.batch_size:
                    self._upsert_relations(edges)
                    edges = []

        if edges:
            self._upsert_relations(edges)

    def expand_documents(
        self,
        seed_documents: list[Document],
        question: str | None = None,
        max_related_chunks: int | None = None,
    ) -> list[Document]:
        if not seed_documents:
            return []

        max_related_chunks = max_related_chunks or config.KG_MAX_EXPANSION_CHUNKS
        selected: dict[str, Document] = {}

        for document in seed_documents:
            metadata = dict(document.metadata or {})
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                source = metadata.get("source", "unknown_source")
                chunk_id = f"{source}::chunk-{len(selected):04d}"
                metadata["chunk_id"] = chunk_id
            selected.setdefault(
                chunk_id,
                Document(page_content=document.page_content, metadata=metadata),
            )

        seed_entities: list[str] = []
        for document in seed_documents:
            metadata = document.metadata or {}
            chunk_id = metadata.get("chunk_id")
            if chunk_id:
                entities = self._entities_for_chunk(chunk_id)
                if entities:
                    seed_entities.extend(entities)
                    continue
            seed_entities.extend(extract_entities(document.page_content))

        question_entities = self._question_entities(question)
        focus_entities = list(dict.fromkeys(seed_entities + question_entities))
        if not focus_entities:
            return list(selected.values())

        candidate_scores = self._candidate_entity_scores(focus_entities)
        if not candidate_scores:
            candidate_scores = {entity: 1 for entity in focus_entities}

        candidate_entities = [
            entity
            for entity, _ in sorted(
                candidate_scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]

        ranked_chunks: list[tuple[int, int, str]] = []
        for row in self._fetch_candidate_chunks(candidate_entities, focus_entities):
            chunk_id = row.get("chunk_id")
            if not chunk_id or chunk_id in selected:
                continue
            chunk_index = self._safe_int(row.get("chunk_index", 0))
            entity_score = int(candidate_scores.get(row.get("entity"), 0))
            overlap = int(row.get("overlap", 0))
            score = entity_score + overlap
            ranked_chunks.append((score, chunk_index, chunk_id))

        ranked_chunks.sort(key=lambda item: (-item[0], item[1], item[2]))

        new_chunk_ids: list[str] = []
        for _, _, chunk_id in ranked_chunks:
            if chunk_id in selected:
                continue
            new_chunk_ids.append(chunk_id)
            if len(new_chunk_ids) >= max_related_chunks:
                break

        if new_chunk_ids:
            fetched = self._fetch_chunks(new_chunk_ids)
            for chunk_id, doc in fetched.items():
                selected.setdefault(chunk_id, doc)

        ordered_documents = sorted(
            selected.values(),
            key=lambda document: (
                self._safe_int(document.metadata.get("chunk_index", 0)),
                str(document.metadata.get("chunk_id", "")),
            ),
        )
        return ordered_documents

    def describe_context(
        self,
        question: str | None,
        documents: list[Document],
        max_entities: int | None = None,
        max_relations: int | None = None,
    ) -> str:
        if not documents:
            return "No graph context available."

        max_entities = max_entities or config.KG_MAX_GRAPH_ENTITIES
        max_relations = max_relations or config.KG_MAX_EXPANSION_CHUNKS

        chunk_ids = [
            (document.metadata or {}).get("chunk_id")
            for document in documents
            if (document.metadata or {}).get("chunk_id")
        ]
        if not chunk_ids:
            return "No graph context available."

        entity_counts: Counter[str] = Counter()
        for row in self._run_read(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
            "WHERE c.id IN $chunk_ids "
            "RETURN e.name AS entity, count(DISTINCT c) AS count",
            {"chunk_ids": chunk_ids},
        ):
            name = row.get("entity")
            count = int(row.get("count", 0))
            if name:
                entity_counts[name] += count

        question_entities = self._question_entities(question)
        for entity in question_entities:
            entity_counts[entity] += 2

        top_entities = [entity for entity, _ in entity_counts.most_common(max_entities)]
        relation_lines: list[str] = []

        if top_entities:
            relation_rows = self._run_read(
                "MATCH (e1:Entity)-[r:CO_OCCURS]-(e2:Entity) "
                "WHERE e1.name IN $entities AND e2.name IN $entities "
                "AND e1.name < e2.name "
                "RETURN e1.name AS left, e2.name AS right, r.weight AS weight "
                "ORDER BY r.weight DESC "
                "LIMIT $limit",
                {"entities": top_entities, "limit": int(max_relations)},
            )
            for row in relation_rows:
                left = row.get("left")
                right = row.get("right")
                weight = row.get("weight")
                if left and right and weight is not None:
                    relation_lines.append(
                        f"{left} <-> {right} (shared chunks: {int(weight)})"
                    )

        lines = [
            f"Question entities: {', '.join(question_entities) if question_entities else 'None detected'}",
            f"Graph entities: {', '.join(top_entities) if top_entities else 'None detected'}",
        ]
        if relation_lines:
            lines.append("Graph relations:")
            lines.extend(f"- {line}" for line in relation_lines)
        return "\n".join(lines)

    def _question_entities(self, question: str | None) -> list[str]:
        if not question:
            return []
        return extract_entities(question)

    def _entities_for_chunk(self, chunk_id: str) -> list[str]:
        rows = self._run_read(
            "MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity) "
            "RETURN e.name AS name",
            {"chunk_id": chunk_id},
        )
        return [row.get("name") for row in rows if row.get("name")]

    def _candidate_entity_scores(self, focus_entities: list[str]) -> dict[str, int]:
        if not focus_entities:
            return {}
        rows = self._run_read(
            "MATCH (e:Entity) WHERE e.name IN $entities "
            "MATCH (e)-[r:CO_OCCURS]-(n:Entity) "
            "RETURN n.name AS entity, sum(r.weight) AS score",
            {"entities": focus_entities},
        )
        scores: dict[str, int] = {}
        for row in rows:
            name = row.get("entity")
            score = row.get("score")
            if name and score is not None:
                scores[name] = int(score)
        return scores

    def _fetch_candidate_chunks(
        self,
        candidate_entities: list[str],
        focus_entities: list[str],
    ) -> list[dict[str, Any]]:
        if not candidate_entities:
            return []
        return self._run_read(
            "MATCH (e:Entity) WHERE e.name IN $entities "
            "MATCH (c:Chunk)-[:MENTIONS]->(e) "
            "OPTIONAL MATCH (c)-[:MENTIONS]->(fe:Entity) "
            "WHERE fe.name IN $focus_entities "
            "WITH c, e, count(DISTINCT fe) AS overlap "
            "RETURN c.id AS chunk_id, c.chunk_index AS chunk_index, "
            "e.name AS entity, overlap",
            {"entities": candidate_entities, "focus_entities": focus_entities},
        )

    def _fetch_chunks(self, chunk_ids: list[str]) -> dict[str, Document]:
        if not chunk_ids:
            return {}
        rows = self._run_read(
            "MATCH (c:Chunk) WHERE c.id IN $chunk_ids "
            "RETURN c.id AS chunk_id, c.text AS text, c.source AS source, "
            "c.chunk_index AS chunk_index, c.metadata AS metadata",
            {"chunk_ids": chunk_ids},
        )
        docs: dict[str, Document] = {}
        for row in rows:
            chunk_id = row.get("chunk_id")
            if not chunk_id:
                continue
            metadata = self._decode_metadata(row.get("metadata"))
            metadata.setdefault("chunk_id", chunk_id)
            if row.get("source") and "source" not in metadata:
                metadata["source"] = row.get("source")
            if row.get("chunk_index") is not None and "chunk_index" not in metadata:
                metadata["chunk_index"] = self._safe_int(row.get("chunk_index"))
            docs[chunk_id] = Document(
                page_content=row.get("text") or "",
                metadata=metadata,
            )
        return docs

    def _upsert_chunks(self, rows: list[dict[str, Any]]) -> None:
        self._run_write(
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {id: row.chunk_id}) "
            "SET c.text = row.text, "
            "c.source = row.source, "
            "c.chunk_index = row.chunk_index, "
            "c.metadata = row.metadata "
            "WITH c, row "
            "UNWIND row.entities AS entity "
            "MERGE (e:Entity {name: entity}) "
            "MERGE (c)-[:MENTIONS]->(e)",
            {"rows": rows},
        )

    def _upsert_relations(self, edges: list[dict[str, Any]]) -> None:
        self._run_write(
            "UNWIND $edges AS row "
            "MATCH (l:Entity {name: row.left}) "
            "MATCH (r:Entity {name: row.right}) "
            "MERGE (l)-[rel:CO_OCCURS]-(r) "
            "SET rel.weight = row.weight",
            {"edges": edges},
        )

    def _run_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        with self._driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def _run_write(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        with self._driver.session(database=self.database) as session:
            session.run(query, parameters or {}).consume()

    def _create_driver(self):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("Neo4j support requires the neo4j package.") from exc
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _encode_metadata(metadata: dict[str, Any]) -> str:
        return json.dumps(metadata, ensure_ascii=True, default=str)

    @staticmethod
    def _decode_metadata(raw: Any) -> dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}


def _graph_store_path(store_path: str | None = None) -> Path:
    return Path(store_path or config.KG_STORE_PATH)


def build_knowledge_graph(documents: list[Document], store_path: str | None = None) -> KnowledgeGraphStore:
    if not documents:
        raise ValueError("Cannot build knowledge graph from an empty document list.")

    graph_store = KnowledgeGraphStore.build(documents)
    graph_store.save(_graph_store_path(store_path))
    return graph_store


def load_knowledge_graph(store_path: str | None = None) -> KnowledgeGraphStore:
    return KnowledgeGraphStore.load(_graph_store_path(store_path))


def get_or_build_knowledge_graph(
    documents: list[Document],
    store_path: str | None = None,
    force_rebuild: bool = False,
) -> KnowledgeGraphStore | Neo4jKnowledgeGraphStore:
    backend = (config.KG_STORE_BACKEND or "json").strip().lower()
    if backend == "neo4j":
        graph_store = Neo4jKnowledgeGraphStore.from_config()
        if graph_store.needs_ingest(force_rebuild):
            clear = force_rebuild and config.NEO4J_CLEAR_ON_REBUILD
            logger.info("Building knowledge graph in Neo4j (clear=%s).", clear)
            graph_store.build(documents, clear=clear)
        else:
            logger.info("Existing Neo4j knowledge graph found; using current store.")
        return graph_store

    if backend not in {"json", "file"}:
        logger.warning(
            "Unknown KG_STORE_BACKEND '%s'; falling back to JSON store.", backend
        )

    path = _graph_store_path(store_path)
    graph_file = path / "knowledge_graph.json"

    if not force_rebuild and graph_file.exists():
        logger.info("Existing knowledge graph found; loading from disk.")
        return load_knowledge_graph(store_path)

    logger.info("No existing knowledge graph found (or rebuild requested); building…")
    return build_knowledge_graph(documents, store_path)