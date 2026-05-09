"""Document processing utilities for MarineRAG-DSS.

Loads documents from the data directory, splits them into chunks,
and prepares them for embedding and indexing.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import List


from langchain_core.documents import Document


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for document in documents:
            text = document.page_content
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(Document(page_content=chunk_text, metadata=dict(document.metadata)))
                if end >= len(text):
                    break
                start += step
        return chunks


class TextLoader:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> list[Document]:
        text = Path(self.file_path).read_text(encoding=self.encoding)
        return [Document(page_content=text, metadata={"source": self.file_path})]


class PyPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - runtime fallback for slim environments
            raise ImportError("PDF loading requires pypdf.") from exc

        reader = PdfReader(self.file_path)
        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            documents.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": self.file_path, "page": page_number},
                )
            )
        return documents


class DocxLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:  # pragma: no cover - runtime fallback for slim environments
            raise ImportError("DOCX loading requires python-docx.") from exc

        doc = DocxDocument(self.file_path)
        parts: list[str] = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        content = "\n".join(parts).strip()
        if not content:
            return []
        return [Document(page_content=content, metadata={"source": self.file_path})]


class DocLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        if shutil.which("antiword") is None and shutil.which("catdoc") is None:
            logger.warning(
                "Skipping DOC '%s' because antiword/catdoc is not available.",
                self.file_path,
            )
            return []
        try:
            import textract
        except ImportError as exc:  # pragma: no cover - runtime fallback for slim environments
            raise ImportError(
                "DOC loading requires textract and an installed .doc parser (antiword or catdoc)."
            ) from exc

        text_bytes = textract.process(self.file_path)
        text = text_bytes.decode("utf-8", errors="ignore").strip()
        if not text:
            return []
        return [Document(page_content=text, metadata={"source": self.file_path})]


class SpreadsheetLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:  # pragma: no cover - runtime fallback for slim environments
            raise ImportError("Spreadsheet loading requires unstructured.") from exc

        elements = partition(filename=self.file_path)
        text_blocks = [str(element).strip() for element in elements if str(element).strip()]
        if not text_blocks:
            return []

        content = "\n\n".join(text_blocks)
        return [Document(page_content=content, metadata={"source": self.file_path})]


class SpreadsheetTableLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime fallback for slim environments
            raise ImportError(
                "Spreadsheet table loading requires pandas with openpyxl/xlrd."
            ) from exc

        file_path = Path(self.file_path)
        ext = file_path.suffix.lower()
        engine = "xlrd" if ext == ".xls" else "openpyxl"

        try:
            sheets = pd.read_excel(
                self.file_path,
                sheet_name=None,
                dtype=str,
                engine=engine,
            )
        except Exception as exc:  # pragma: no cover - file-specific errors
            raise RuntimeError(
                f"Failed to read spreadsheet '{self.file_path}': {exc}"
            ) from exc

        documents: list[Document] = []
        for sheet_name, frame in sheets.items():
            if frame is None:
                continue
            frame = frame.dropna(how="all")
            if frame.empty:
                continue
            frame = frame.fillna("")
            columns = [self._clean_column_name(col) for col in frame.columns]
            data_frame = frame

            if self._needs_header_rebuild(columns):
                header_idx = self._find_header_row(frame)
                if header_idx is not None:
                    header_pos = frame.index.get_loc(header_idx)
                    if isinstance(header_pos, slice):
                        header_pos = header_pos.start or 0
                    elif not isinstance(header_pos, int):
                        header_pos = int(header_pos[0]) if len(header_pos) else 0
                    header_row = frame.iloc[header_pos].to_list()
                    subheader_row = (
                        frame.iloc[header_pos + 1].to_list()
                        if header_pos + 1 < len(frame)
                        else []
                    )
                    columns = self._combine_header_rows(header_row, subheader_row)
                    data_frame = frame.iloc[header_pos + 2 :].copy()
                else:
                    columns = [f"col_{i + 1}" for i in range(frame.shape[1])]

            for row_number, row in enumerate(
                data_frame.itertuples(index=False, name=None),
                start=1,
            ):
                row_values: dict[str, str] = {}
                for col, value in zip(columns, row):
                    if not col:
                        continue
                    value_str = str(value).strip()
                    if not value_str or value_str.lower() == "nan":
                        continue
                    row_values[col] = value_str

                if not row_values:
                    continue

                if "Thông số" in columns and not str(row_values.get("Thông số", "")).strip():
                    continue

                stt_value = str(row_values.get("STT", "")).strip().lower()
                if "giá trị trọng số" in stt_value or "gía trị trọng số" in stt_value:
                    continue

                content = " | ".join(
                    f"{col}: {value}" for col, value in row_values.items()
                )
                safe_sheet = self._safe_sheet_name(sheet_name)
                metadata = {
                    "source": self.file_path,
                    "sheet": str(sheet_name),
                    "row_index": row_number,
                    "row_data": row_values,
                    "doc_type": "table_row",
                    "chunk_id": (
                        f"{self.file_path}::sheet-{safe_sheet}::row-{row_number:04d}"
                    ),
                }
                documents.append(Document(page_content=content, metadata=metadata))

        return documents

    @staticmethod
    def _find_header_row(frame) -> int | None:
        for index, row in frame.iterrows():
            for value in row.tolist():
                if str(value).strip().lower() == "stt":
                    return int(index)
        return None

    @staticmethod
    def _needs_header_rebuild(columns: list[str]) -> bool:
        normalized = {col.strip().lower() for col in columns if col and col.strip()}
        has_expected = "stt" in normalized and (
            "thông số" in normalized or "thong so" in normalized
        )
        return not has_expected

    @staticmethod
    def _combine_header_rows(header_row: list, subheader_row: list) -> list[str]:
        columns: list[str] = []
        for idx, base in enumerate(header_row):
            base_text = str(base).strip()
            sub_text = str(subheader_row[idx]).strip() if idx < len(subheader_row) else ""
            if idx <= 2:
                name = base_text or f"col_{idx + 1}"
            else:
                name = sub_text or base_text or f"col_{idx + 1}"
            columns.append(name)
        return columns

    @staticmethod
    def _clean_column_name(name: object) -> str:
        text = str(name).strip()
        if not text or text.lower().startswith("unnamed"):
            return ""
        return text

    @staticmethod
    def _safe_sheet_name(name: object) -> str:
        text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
        return text or "Sheet"


from src.config import config

logger = logging.getLogger(__name__)


def load_documents(
    data_dir: str | None = None,
    include_spreadsheets: bool = True,
) -> list[Document]:
    """Load all supported documents from the given directory.

    Args:
        data_dir: Path to the directory containing documents.
                  Defaults to config.DATA_DIR.
        include_spreadsheets: Whether to include spreadsheets as plain text.

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

    # Load DOCX files
    for file_path in data_path.rglob("*.docx"):
        try:
            loader = DocxLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", str(file_path))
            documents.extend(docs)
            logger.info("Loaded %d chunks from DOCX '%s'", len(docs), file_path)
        except Exception:
            logger.exception("Failed to load DOCX '%s'", file_path)

    # Load DOC files
    for file_path in data_path.rglob("*.doc"):
        try:
            loader = DocLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", str(file_path))
            documents.extend(docs)
            logger.info("Loaded %d chunks from DOC '%s'", len(docs), file_path)
        except Exception:
            logger.exception("Failed to load DOC '%s'", file_path)

    # Load Excel files (.xls/.xlsx)
    if include_spreadsheets:
        for ext in ("*.xls", "*.xlsx"):
            for file_path in data_path.rglob(ext):
                try:
                    loader = SpreadsheetLoader(str(file_path))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.setdefault("source", str(file_path))
                    documents.extend(docs)
                    logger.info(
                        "Loaded %d chunks from spreadsheet '%s'", len(docs), file_path
                    )
                except Exception:
                    logger.exception("Failed to load spreadsheet '%s'", file_path)

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def load_table_documents(data_dir: str | None = None) -> list[Document]:
    """Load spreadsheet rows as structured table documents.

    Args:
        data_dir: Path to the directory containing spreadsheets.
                  Defaults to config.DATA_DIR.

    Returns:
        A list of Document objects, one per row.
    """
    data_dir = data_dir or config.DATA_DIR
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning("Data directory '%s' does not exist.", data_dir)
        return []

    documents: list[Document] = []
    for ext in ("*.xls", "*.xlsx"):
        for file_path in data_path.rglob(ext):
            try:
                loader = SpreadsheetTableLoader(str(file_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata.setdefault("source", str(file_path))
                documents.extend(docs)
                logger.info(
                    "Loaded %d table rows from spreadsheet '%s'", len(docs), file_path
                )
            except Exception:
                logger.exception("Failed to load spreadsheet '%s'", file_path)

    logger.info("Total table rows loaded: %d", len(documents))
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

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown_source")
        chunk.metadata.setdefault("chunk_index", index)
        chunk.metadata.setdefault("chunk_id", f"{source}::chunk-{index:04d}")

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
    include_spreadsheets: bool = True,
) -> list[Document]:
    """Convenience function: load documents and split into chunks.

    Args:
        data_dir: Directory containing source documents.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        include_spreadsheets: Whether to include spreadsheets as plain text.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    documents = load_documents(data_dir, include_spreadsheets=include_spreadsheets)
    if not documents:
        logger.warning("No documents found; returning empty list.")
        return []
    return split_documents(documents, chunk_size, chunk_overlap)
