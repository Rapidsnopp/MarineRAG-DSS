"""Structured table store utilities for spreadsheet data."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document

from src.config import config

_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SOURCE_TOKEN_WEIGHT = 3
_MAX_EXPANDED_SHEET_ROWS = 50

_STOPWORDS = {
    "and",
    "or",
    "the",
    "of",
    "to",
    "in",
    "for",
    "a",
    "an",
    "on",
    "at",
    "with",
    "by",
    "from",
    "va",
    "cua",
    "trong",
    "cho",
    "den",
    "tu",
    "voi",
    "la",
    "nam",
    "thang",
    "ngay",
    "bao",
    "cao",
    "so",
    "lieu",
    "ve",
    "theo",
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    return without_marks.replace("\u0111", "d").replace("\u0110", "D")


def _normalize_text(text: str) -> str:
    text = _strip_accents(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = _TOKEN_PATTERN.findall(normalized)
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _extract_years(text: str) -> list[str]:
    return list(dict.fromkeys(_YEAR_PATTERN.findall(text)))


def _should_expand_sheet(query_tokens: set[str]) -> bool:
    return "nguong" in query_tokens or "threshold" in query_tokens


@dataclass
class _IndexedRow:
    document: Document
    tokens: set[str]
    column_tokens: set[str]
    source_tokens: set[str]
    normalized_text: str


class TableStore:
    """In-memory table row store with simple structured retrieval."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._indexed = [self._index_document(doc) for doc in documents]

    def _index_document(self, document: Document) -> _IndexedRow:
        metadata = document.metadata or {}
        row_data = metadata.get("row_data") or {}
        combined_text = " ".join(
            item
            for item in [
                document.page_content,
                str(metadata.get("source", "")),
                str(metadata.get("sheet", "")),
            ]
            if item
        )
        source_text = " ".join(
            item
            for item in [
                str(metadata.get("source", "")),
                str(metadata.get("sheet", "")),
            ]
            if item
        )
        normalized_text = _normalize_text(combined_text)
        tokens = set(_tokenize(normalized_text))
        column_tokens = set(_tokenize(" ".join(str(col) for col in row_data.keys())))
        source_tokens = set(_tokenize(source_text))
        return _IndexedRow(
            document=document,
            tokens=tokens,
            column_tokens=column_tokens,
            source_tokens=source_tokens,
            normalized_text=normalized_text,
        )

    def query(
        self,
        query: str,
        k: int | None = None,
        min_score: int | None = None,
    ) -> list[Document]:
        if not self.documents:
            return []

        k = k or config.TABLE_RETRIEVAL_K
        min_score = (
            config.TABLE_STRUCTURED_MIN_SCORE if min_score is None else min_score
        )

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        years = _extract_years(query)
        numeric_tokens = {token for token in query_tokens if token.isdigit()}

        scored: list[tuple[int, int, Document]] = []
        for index, entry in enumerate(self._indexed):
            if years and not any(year in entry.normalized_text for year in years):
                continue

            overlap = len(entry.tokens.intersection(query_tokens))
            numeric_overlap = len(entry.tokens.intersection(numeric_tokens))
            column_overlap = len(entry.column_tokens.intersection(query_tokens))
            source_overlap = len(entry.source_tokens.intersection(query_tokens))
            score = overlap + numeric_overlap + column_overlap + source_overlap * _SOURCE_TOKEN_WEIGHT

            if score < min_score:
                continue
            scored.append((score, index, entry.document))

        scored.sort(key=lambda item: (-item[0], item[1]))
        if _should_expand_sheet(query_tokens) and scored:
            top_doc = scored[0][2]
            top_meta = top_doc.metadata or {}
            top_source = top_meta.get("source")
            top_sheet = top_meta.get("sheet")
            if top_source:
                expanded = []
                for entry in self._indexed:
                    meta = entry.document.metadata or {}
                    if meta.get("source") != top_source:
                        continue
                    if top_sheet and meta.get("sheet") != top_sheet:
                        continue
                    expanded.append(entry.document)
                expanded.sort(key=self._row_index_sort_key)
                cap = max(k, _MAX_EXPANDED_SHEET_ROWS)
                return expanded[:cap]
        return [doc for _, _, doc in scored[:k]]

    @staticmethod
    def _row_index_sort_key(document: Document) -> tuple[int, str]:
        metadata = document.metadata or {}
        try:
            row_index = int(metadata.get("row_index"))
        except (TypeError, ValueError):
            row_index = 1_000_000_000
        chunk_id = str(metadata.get("chunk_id", ""))
        return (row_index, chunk_id)


class MySQLTableStore:
    """MySQL-backed table row store with simple structured retrieval."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        table: str,
    ) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = self._safe_identifier(table)
        self._schema_ready = False

    @classmethod
    def from_config(cls) -> "MySQLTableStore":
        if not config.TABLE_MYSQL_USER:
            raise ValueError("TABLE_MYSQL_USER is not set.")
        if not config.TABLE_MYSQL_DB:
            raise ValueError("TABLE_MYSQL_DB is not set.")
        return cls(
            host=config.TABLE_MYSQL_HOST,
            port=config.TABLE_MYSQL_PORT,
            database=config.TABLE_MYSQL_DB,
            user=config.TABLE_MYSQL_USER,
            password=config.TABLE_MYSQL_PASSWORD,
            table=config.TABLE_MYSQL_TABLE,
        )

    def needs_ingest(self, force_rebuild: bool) -> bool:
        if force_rebuild:
            return True
        return self.is_empty()

    def is_empty(self) -> bool:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table}")
                return cursor.fetchone()[0] == 0

    def ingest_documents(
        self,
        documents: Iterable[Document],
        force_rebuild: bool = False,
    ) -> None:
        docs = list(documents)
        if not docs:
            return

        self.ensure_schema()
        if force_rebuild:
            self.clear()

        rows = []
        for document in docs:
            metadata = document.metadata or {}
            row_key = self._row_key(metadata)
            row_data = metadata.get("row_data") or {}
            row_data_json = json.dumps(row_data, ensure_ascii=True)
            content = document.page_content or ""
            normalized_text = self._normalized_text(content, metadata)
            normalized_columns = _normalize_text(" ".join(row_data.keys()))
            rows.append(
                (
                    row_key,
                    metadata.get("source"),
                    metadata.get("sheet"),
                    metadata.get("row_index"),
                    row_data_json,
                    content,
                    normalized_text,
                    normalized_columns,
                )
            )

        insert_sql = (
            f"INSERT INTO {self.table} "
            "(row_key, source, sheet, row_index, row_data, content, "
            "normalized_text, normalized_columns) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "source=VALUES(source), sheet=VALUES(sheet), row_index=VALUES(row_index), "
            "row_data=VALUES(row_data), content=VALUES(content), "
            "normalized_text=VALUES(normalized_text), "
            "normalized_columns=VALUES(normalized_columns)"
        )

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, rows)
            conn.commit()

    def query(
        self,
        query: str,
        k: int | None = None,
        min_score: int | None = None,
    ) -> list[Document]:
        self.ensure_schema()
        k = k or config.TABLE_RETRIEVAL_K
        min_score = (
            config.TABLE_STRUCTURED_MIN_SCORE if min_score is None else min_score
        )

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        years = _extract_years(query)
        numeric_tokens = {token for token in query_tokens if token.isdigit()}

        tokens = sorted(query_tokens, key=len, reverse=True)
        token_limit = max(1, config.TABLE_MYSQL_TOKEN_LIMIT)
        tokens = tokens[:token_limit]

        where_clauses = []
        params: list[str | int] = []
        for year in years:
            where_clauses.append("normalized_text LIKE %s")
            params.append(f"%{year}%")

        token_clauses = []
        for token in tokens:
            token_clauses.append("normalized_text LIKE %s")
            params.append(f"%{token}%")
            token_clauses.append("normalized_columns LIKE %s")
            params.append(f"%{token}%")
        if token_clauses:
            where_clauses.append("(" + " OR ".join(token_clauses) + ")")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        candidate_limit = max(k * 20, config.TABLE_MYSQL_CANDIDATE_LIMIT)
        params.append(candidate_limit)

        select_sql = (
            "SELECT row_key, source, sheet, row_index, row_data, content, "
            "normalized_text, normalized_columns "
            f"FROM {self.table} WHERE {where_sql} LIMIT %s"
        )

        candidates: list[dict] = []
        with self._connect() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(select_sql, params)
                candidates = cursor.fetchall()

        scored: list[tuple[int, int, Document]] = []
        for index, row in enumerate(candidates):
            normalized_text = row.get("normalized_text") or ""
            row_data = row.get("row_data") or {}
            if isinstance(row_data, str):
                try:
                    row_data = json.loads(row_data)
                except json.JSONDecodeError:
                    row_data = {}

            tokens_set = set(_tokenize(normalized_text))
            column_tokens = set(_tokenize(" ".join(row_data.keys())))

            overlap = len(tokens_set.intersection(query_tokens))
            numeric_overlap = len(tokens_set.intersection(numeric_tokens))
            column_overlap = len(column_tokens.intersection(query_tokens))
            source_text = " ".join(
                item
                for item in [
                    str(row.get("source", "")),
                    str(row.get("sheet", "")),
                ]
                if item
            )
            source_tokens = set(_tokenize(source_text))
            source_overlap = len(source_tokens.intersection(query_tokens))
            score = overlap + numeric_overlap + column_overlap + source_overlap * _SOURCE_TOKEN_WEIGHT
            if score < min_score:
                continue

            metadata = {
                "source": row.get("source"),
                "sheet": row.get("sheet"),
                "row_index": row.get("row_index"),
                "row_data": row_data,
                "doc_type": "table_row",
                "chunk_id": row.get("row_key"),
            }
            doc = Document(page_content=row.get("content") or "", metadata=metadata)
            scored.append((score, index, doc))

        scored.sort(key=lambda item: (-item[0], item[1]))
        if _should_expand_sheet(query_tokens) and scored:
            top_doc = scored[0][2]
            top_meta = top_doc.metadata or {}
            top_source = top_meta.get("source")
            top_sheet = top_meta.get("sheet")
            if top_source:
                expanded = self._fetch_rows_for_source(top_source, top_sheet, max(k, _MAX_EXPANDED_SHEET_ROWS))
                if expanded:
                    return expanded
        return [doc for _, _, doc in scored[:k]]

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.table} ("
                    "row_key VARCHAR(512) PRIMARY KEY, "
                    "source TEXT, "
                    "sheet VARCHAR(255), "
                    "row_index INT, "
                    "row_data JSON, "
                    "content LONGTEXT, "
                    "normalized_text LONGTEXT, "
                    "normalized_columns TEXT, "
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
                    ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
                )
                try:
                    cursor.execute(
                        f"CREATE FULLTEXT INDEX idx_fulltext ON {self.table} "
                        "(normalized_text, normalized_columns)"
                    )
                except Exception:
                    pass
            conn.commit()
        self._schema_ready = True

    def clear(self) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {self.table}")
            conn.commit()

    def _connect(self):
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "MySQL table store requires mysql-connector-python."
            ) from exc

        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            autocommit=False,
            charset="utf8mb4",
            use_unicode=True,
        )

    def _fetch_rows_for_source(
        self,
        source: str,
        sheet: str | None,
        limit: int,
    ) -> list[Document]:
        where_sql = "source = %s"
        params: list[str | int] = [source]
        if sheet:
            where_sql += " AND sheet = %s"
            params.append(sheet)
        params.append(limit)

        select_sql = (
            "SELECT row_key, source, sheet, row_index, row_data, content "
            f"FROM {self.table} WHERE {where_sql} ORDER BY row_index ASC LIMIT %s"
        )

        with self._connect() as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(select_sql, params)
                rows = cursor.fetchall()

        return self._rows_to_documents(rows)

    @staticmethod
    def _rows_to_documents(rows: list[dict]) -> list[Document]:
        documents: list[Document] = []
        for row in rows:
            row_data = row.get("row_data") or {}
            if isinstance(row_data, str):
                try:
                    row_data = json.loads(row_data)
                except json.JSONDecodeError:
                    row_data = {}

            metadata = {
                "source": row.get("source"),
                "sheet": row.get("sheet"),
                "row_index": row.get("row_index"),
                "row_data": row_data,
                "doc_type": "table_row",
                "chunk_id": row.get("row_key"),
            }
            documents.append(
                Document(page_content=row.get("content") or "", metadata=metadata)
            )
        return documents

    @staticmethod
    def _safe_identifier(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
            raise ValueError(
                "MySQL table name must contain only letters, numbers, and underscores."
            )
        return name

    @staticmethod
    def _row_key(metadata: dict) -> str:
        raw_key = metadata.get("chunk_id")
        if not raw_key:
            source = metadata.get("source", "")
            sheet = metadata.get("sheet", "")
            row_index = metadata.get("row_index", "")
            raw_key = f"{source}::sheet-{sheet}::row-{row_index}"
        raw_key = str(raw_key)
        if len(raw_key) <= 512:
            return raw_key
        digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()
        return f"row_{digest}"

    @staticmethod
    def _normalized_text(content: str, metadata: dict) -> str:
        combined_text = " ".join(
            item
            for item in [
                content,
                str(metadata.get("source", "")),
                str(metadata.get("sheet", "")),
            ]
            if item
        )
        return _normalize_text(combined_text)


def format_table_documents(documents: list[Document]) -> str:
    if not documents:
        return "No structured table data available."

    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown")
        sheet = metadata.get("sheet")
        row_index = metadata.get("row_index")
        header = f"[{index}] {source}"
        details = []
        if sheet:
            details.append(f"sheet: {sheet}")
        if row_index is not None:
            details.append(f"row: {row_index}")
        if details:
            header += " | " + ", ".join(details)
        lines.append(header)

        row_data = metadata.get("row_data") or {}
        if row_data:
            row_parts = [
                f"{col}: {value}"
                for col, value in row_data.items()
                if str(value).strip()
            ]
            lines.append(" | ".join(row_parts))
        else:
            lines.append(doc.page_content)

    return "\n".join(lines)
