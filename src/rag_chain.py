"""RAG chain construction for MarineRAG-DSS.

Builds a conversational retrieval-augmented generation (RAG) chain
that combines a vector-store retriever, optional structured table
lookup, and a chat LLM to answer questions about marine resource
management.

The chain is built using LangChain Expression Language (LCEL):
  input → retrieve context → format prompt → LLM → parse output
"""

from __future__ import annotations

import logging
import re
import unicodedata


from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import config
from src.kg_store import KnowledgeGraphStore, Neo4jKnowledgeGraphStore
from src.table_store import MySQLTableStore, TableStore, format_table_documents
from src.web_search import search_web, web_results_to_documents

logger = logging.getLogger(__name__)

_YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_MONTH_YEAR_PATTERN = re.compile(r"\b(0?[1-9]|1[0-2])\s*[/.-]\s*((?:19|20)\d{2})\b")
_DAY_MONTH_PATTERN = re.compile(r"\b(\d{1,2})\s*[-./]\s*(\d{1,2})(?:\s*[-./]\s*(\d{1,4}))?\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "bao",
    "cao",
    "cho",
    "cua",
    "các",
    "cac",
    "co",
    "có",
    "den",
    "đến",
    "duoc",
    "được",
    "for",
    "in",
    "la",
    "là",
    "nam",
    "năm",
    "ngay",
    "ngày",
    "noi",
    "nội",
    "of",
    "o",
    "ở",
    "qua",
    "the",
    "thang",
    "tháng",
    "thi",
    "thị",
    "to",
    "trong",
    "tu",
    "từ",
    "va",
    "và",
    "ve",
    "về",
    "voi",
    "với",
}

_ENV_QUERY_HINTS = (
    "quan trac",
    "moi truong",
    "chi so",
    "chat luong nuoc",
    "khuyen cao",
    "nhan xet",
    "ph",
    "do",
    "nh3",
    "h2s",
    "po43",
    "tss",
)

_GUIDE_QUERY_HINTS = (
    "huong dan",
    "hướng dẫn",
    "su dung",
    "sử dụng",
    "cach dung",
    "cách dùng",
    "dang nhap",
    "đăng nhập",
    "login",
    "sign in",
    "signin",
    "how to use",
    "user guide",
    "manual",
)

_THRESHOLD_QUERY_HINTS = (
    "nguong",
    "threshold",
    "sinh truong",
    "phu hop",
    "thong so",
    "chi tieu",
)

_GROWTH_THRESHOLD_SOURCE_HINTS = (
    "/growth_threshold/",
    "growth_threshold_",
    "chuyen gia thong nhat",
    "nguong",
)

_SPECIES_QUERY_HINTS: dict[str, tuple[str, ...]] = {
    "tree_worm": (
        "cay ban chua",
        "ban chua",
        "mangrove worm",
        "tree worm",
    ),
    "oyster": (
        "hau",
        "oyster",
    ),
    "mullet": (
        "ca gio",
        "mullet",
    ),
}


def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page contents for use in the prompt."""
    lines: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        chunk_id = doc.metadata.get("chunk_id")
        prefix = f"[{index}] {source}"
        if chunk_id:
            prefix += f" | {chunk_id}"
        lines.append(f"{prefix}\n{doc.page_content}")
    return "\n\n".join(lines)


def _extract_years(text: str) -> list[str]:
    return list(dict.fromkeys(_YEAR_PATTERN.findall(text)))


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def _normalize_temporal_text(text: str) -> str:
    cleaned = _strip_accents(text or "").lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _tokenize_lexical(text: str) -> set[str]:
    normalized = _normalize_temporal_text(text)
    tokens = _TOKEN_PATTERN.findall(normalized)
    return {
        token
        for token in tokens
        if len(token) > 1 and token not in _LEXICAL_STOPWORDS
    }


def _has_environment_focus(query_text: str) -> bool:
    normalized = _normalize_temporal_text(query_text)
    return any(hint in normalized for hint in _ENV_QUERY_HINTS)


def _lexical_rank_docs(docs: list[Document], query: str) -> list[Document]:
    if not docs:
        return docs

    query_text = _normalize_temporal_text(query)
    query_tokens = _tokenize_lexical(query_text)
    env_focus = _has_environment_focus(query_text)

    scored: list[tuple[int, int, Document]] = []
    for index, doc in enumerate(docs):
        source = str((doc.metadata or {}).get("source", ""))
        content = doc.page_content or ""
        doc_tokens = _tokenize_lexical(f"{source} {content}")
        overlap = len(query_tokens.intersection(doc_tokens))
        score = overlap

        normalized_doc = _normalize_temporal_text(content)
        if env_focus and any(hint in normalized_doc for hint in _ENV_QUERY_HINTS):
            score += 5

        scored.append((score, index, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored]


def _extract_month_year_filters(
    text: str,
    years: list[str] | None = None,
) -> list[tuple[int, int | None]]:
    normalized = _normalize_temporal_text(text)
    parsed_years = years or _extract_years(normalized)
    fallback_year = int(parsed_years[0]) if len(parsed_years) == 1 else None

    filters: list[tuple[int, int | None]] = []
    seen: set[tuple[int, int | None]] = set()

    def _add(month: int, year: int | None) -> None:
        if month < 1 or month > 12:
            return
        item = (month, year)
        if item in seen:
            return
        seen.add(item)
        filters.append(item)

    for month_text, year_text in _MONTH_YEAR_PATTERN.findall(normalized):
        _add(int(month_text), int(year_text))

    for match in re.findall(r"\bthang\s*(\d{1,2})(?:\s*nam\s*((?:19|20)\d{2}))?", normalized):
        month_text, year_text = match
        month = int(month_text)
        year = int(year_text) if year_text else fallback_year
        _add(month, year)

    for match in re.findall(r"\bthang\s*(\d{1,2})\b", normalized):
        month = int(match)
        existing_with_same_month = any(item[0] == month for item in filters)
        if existing_with_same_month:
            continue
        _add(month, fallback_year)

    return filters


def _extract_source_month_year(source: str) -> tuple[int | None, int | None]:
    normalized = _normalize_temporal_text(source).replace("\\", "/")

    year_matches = _extract_years(normalized)
    year = int(year_matches[0]) if year_matches else None

    month: int | None = None
    detailed_match = re.search(
        r"\bngay\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?:\s*[-./]\s*(\d{1,4}))?",
        normalized,
    )
    generic_match = _DAY_MONTH_PATTERN.search(normalized)
    match = detailed_match or generic_match
    if match:
        candidate_month = int(match.group(2))
        if 1 <= candidate_month <= 12:
            month = candidate_month
        candidate_year = match.group(3)
        if candidate_year and len(candidate_year) == 4:
            year = int(candidate_year)

    return month, year


def _doc_matches_month_year(doc: Document, month_filters: list[tuple[int, int | None]]) -> bool:
    if not month_filters:
        return True

    source = str((doc.metadata or {}).get("source", ""))
    month, year = _extract_source_month_year(source)
    if month is None:
        return False

    for expected_month, expected_year in month_filters:
        if month != expected_month:
            continue
        if expected_year is None:
            return True
        if year is not None and year == expected_year:
            return True
    return False


def _filter_docs_by_month_year(
    docs: list[Document],
    month_filters: list[tuple[int, int | None]],
) -> list[Document]:
    if not month_filters:
        return docs
    return [doc for doc in docs if _doc_matches_month_year(doc, month_filters)]


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    deduped: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        metadata = doc.metadata or {}
        doc_id = str(metadata.get("chunk_id") or "")
        if not doc_id:
            source = str(metadata.get("source", ""))
            doc_id = f"{source}::{hash(doc.page_content)}"
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc)
    return deduped


def _is_short_followup(question: str) -> bool:
    tokens = [token for token in question.strip().split() if token]
    if len(tokens) <= 3:
        return True
    lowered = question.strip().lower()
    if re.fullmatch(r"\d{4}", lowered):
        return True
    if lowered.startswith("nam "):
        return True
    return False


def _last_user_message(chat_history: list) -> str:
    for message in reversed(chat_history or []):
        message_type = getattr(message, "type", "")
        if message_type == "human":
            return getattr(message, "content", "") or ""
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "") or ""
    return ""


def _year_in_source(doc: Document, years: list[str]) -> bool:
    source = str(doc.metadata.get("source", ""))
    return any(year in source for year in years)


def _is_guide_source(doc: Document) -> bool:
    source = str(doc.metadata.get("source", "")).replace("\\", "/").lower()
    return "/user_guide/" in source or source.endswith("user_guide.pdf")


def _is_guide_query(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(hint in lowered for hint in _GUIDE_QUERY_HINTS)


def _is_growth_threshold_query(text: str) -> bool:
    normalized = _normalize_temporal_text(text)
    return any(hint in normalized for hint in _THRESHOLD_QUERY_HINTS)


def _detect_growth_species_keys(text: str) -> set[str]:
    normalized = _normalize_temporal_text(text)
    matched: set[str] = set()
    for species_key, hints in _SPECIES_QUERY_HINTS.items():
        if any(hint in normalized for hint in hints):
            matched.add(species_key)
    return matched


def _doc_search_text(doc: Document) -> str:
    metadata = doc.metadata or {}
    parts = [
        str(metadata.get("source", "")),
        str(metadata.get("sheet", "")),
        doc.page_content or "",
    ]
    row_data = metadata.get("row_data")
    if isinstance(row_data, dict):
        parts.extend(str(value) for value in row_data.values())
    merged = " ".join(part for part in parts if part)
    return _normalize_temporal_text(merged)


def _doc_matches_growth_species(doc: Document, species_keys: set[str]) -> bool:
    if not species_keys:
        return True

    doc_text = _doc_search_text(doc)
    for species_key in species_keys:
        hints = _SPECIES_QUERY_HINTS.get(species_key, ())
        if any(hint in doc_text for hint in hints):
            return True
    return False


def _is_growth_threshold_source(doc: Document) -> bool:
    source = str((doc.metadata or {}).get("source", ""))
    if not source:
        return False
    normalized_source = _normalize_temporal_text(source).replace("\\", "/")
    return any(hint in normalized_source for hint in _GROWTH_THRESHOLD_SOURCE_HINTS)


def _filter_growth_threshold_table_docs(docs: list[Document], query: str) -> list[Document]:
    if not docs:
        return []

    ranked = _lexical_rank_docs(_dedupe_docs(docs), query)
    threshold_docs = [doc for doc in ranked if _is_growth_threshold_source(doc)]
    filtered = threshold_docs or ranked

    species_keys = _detect_growth_species_keys(query)
    if species_keys:
        species_docs = [
            doc for doc in filtered if _doc_matches_growth_species(doc, species_keys)
        ]
        if species_docs:
            filtered = species_docs

    return _dedupe_docs(filtered)


def _filter_growth_threshold_context_docs(
    docs: list[Document],
    query: str,
    has_threshold_table_docs: bool,
) -> list[Document]:
    if not docs:
        return []

    ranked = _lexical_rank_docs(_dedupe_docs(docs), query)
    threshold_docs = [doc for doc in ranked if _is_growth_threshold_source(doc)]
    if threshold_docs:
        ranked = threshold_docs
    elif has_threshold_table_docs:
        return []

    species_keys = _detect_growth_species_keys(query)
    if species_keys:
        species_docs = [doc for doc in ranked if _doc_matches_growth_species(doc, species_keys)]
        if species_docs:
            ranked = species_docs

    return _dedupe_docs(ranked)


def _boost_docs_by_year(docs: list[Document], years: list[str]) -> list[Document]:
    if not years:
        return docs
    with_year = [doc for doc in docs if _year_in_source(doc, years)]
    if not with_year:
        return docs
    without_year = [doc for doc in docs if doc not in with_year]
    return with_year + without_year


def _retrieve_year_docs(
    retriever: VectorStoreRetriever,
    query: str,
    years: list[str],
    k: int,
) -> list[Document]:
    store = getattr(retriever, "vectorstore", None) or getattr(retriever, "store", None)
    if store is None or not hasattr(store, "similarity_search"):
        return []

    candidate_k = max(k * 4, k)
    candidates = store.similarity_search(query, k=candidate_k)
    year_docs = [doc for doc in candidates if _year_in_source(doc, years)]
    return year_docs[:k]


def _retrieve_guide_docs(
    retriever: VectorStoreRetriever,
    query: str,
    k: int,
) -> list[Document]:
    store = getattr(retriever, "vectorstore", None) or getattr(retriever, "store", None)
    if store is None or not hasattr(store, "similarity_search"):
        return []

    candidate_k = max(k * 10, 30)
    candidates = store.similarity_search(query, k=candidate_k)
    guide_docs = [doc for doc in candidates if _is_guide_source(doc)]
    return guide_docs[:k]


def _retrieve_month_docs(
    retriever: VectorStoreRetriever,
    query: str,
    month_filters: list[tuple[int, int | None]],
    k: int,
) -> list[Document]:
    if not month_filters:
        return []

    store = getattr(retriever, "vectorstore", None) or getattr(retriever, "store", None)
    if store is None or not hasattr(store, "similarity_search"):
        return []

    candidate_k = max(k * 12, 24)
    candidates = store.similarity_search(query, k=candidate_k)
    filtered = _filter_docs_by_month_year(candidates, month_filters)
    if filtered:
        ranked = _lexical_rank_docs(_dedupe_docs(filtered), query)
        return ranked[:k]

    temporal_candidates: list[Document] = []
    for month, year in month_filters:
        temporal_query = f"{query} thang {month}" + (f" nam {year}" if year else "")
        temporal_candidates.extend(store.similarity_search(temporal_query, k=candidate_k))

    filtered_temporal = _filter_docs_by_month_year(temporal_candidates, month_filters)
    ranked_temporal = _lexical_rank_docs(_dedupe_docs(filtered_temporal), query)
    return ranked_temporal[:k]


def build_rag_chain(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    knowledge_graph: KnowledgeGraphStore | Neo4jKnowledgeGraphStore | None = None,
    system_prompt: str | None = None,
    max_graph_chunks: int | None = None,
    table_store: TableStore | MySQLTableStore | None = None,
    table_k: int | None = None,
) -> Runnable:
    """Assemble a RAG chain from an LLM and a retriever using LCEL.

    The chain follows a *retrieve-then-read* pattern:
    1. The retriever fetches the ``k`` most relevant chunks for the query.
    2. The LLM generates an answer conditioned on those chunks.

    Args:
        llm: Chat language model (OpenAI or Ollama).
        retriever: Document retriever backed by the vector store.
        knowledge_graph: Optional graph store used to expand retrieved chunks.
        system_prompt: Optional override for the system/context prompt.
                       Must contain the ``{context}`` placeholder.
        max_graph_chunks: Maximum number of graph-expanded chunks to add.

    Returns:
        A LangChain ``Runnable`` that accepts
        ``{"input": <question>, "chat_history": [...]}`` (chat_history optional)
        and returns ``{"answer": <response>, "context": [<docs>]}``.
    """
    system_prompt = system_prompt or config.RAG_SYSTEM_PROMPT
    max_graph_chunks = max_graph_chunks or config.KG_MAX_EXPANSION_CHUNKS

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
        ]
    )

    answer_chain = prompt | llm | StrOutputParser()

    def full_rag(inputs: dict) -> dict:
        question = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        question_for_retrieval = question
        if _is_short_followup(question):
            previous_question = _last_user_message(chat_history)
            if previous_question:
                question_for_retrieval = f"{previous_question} {question}".strip()

        guide_query = _is_guide_query(question_for_retrieval)
        threshold_query = _is_growth_threshold_query(question_for_retrieval)

        docs = retriever.invoke(question_for_retrieval)
        years = _extract_years(question_for_retrieval)
        month_filters = _extract_month_year_filters(question_for_retrieval, years)

        if month_filters:
            docs = _filter_docs_by_month_year(docs, month_filters)
            fallback_k = max(getattr(retriever, "k", config.RETRIEVAL_K) * 3, 12)
            month_docs = _retrieve_month_docs(
                retriever,
                question_for_retrieval,
                month_filters,
                fallback_k,
            )

            if month_docs:
                docs = _lexical_rank_docs(_dedupe_docs(docs + month_docs), question_for_retrieval)
                docs = docs[:fallback_k]
            else:
                docs = _lexical_rank_docs(_dedupe_docs(docs), question_for_retrieval)

        docs = _boost_docs_by_year(docs, years)

        if docs:
            if guide_query:
                docs = [doc for doc in docs if _is_guide_source(doc)]
            else:
                non_guide_docs = [doc for doc in docs if not _is_guide_source(doc)]
                if non_guide_docs:
                    docs = non_guide_docs

        if guide_query:
            fallback_k = max(getattr(retriever, "k", config.RETRIEVAL_K) * 4, 12)
            extra_guide_docs = _retrieve_guide_docs(
                retriever,
                question_for_retrieval,
                fallback_k,
            )
            docs = _dedupe_docs(docs + extra_guide_docs)
            docs = [doc for doc in docs if _is_guide_source(doc)]

        table_docs: list[Document] = []
        if table_store is not None and not guide_query:
            table_docs = table_store.query(
                question_for_retrieval,
                k=table_k or config.TABLE_RETRIEVAL_K,
            )
            if month_filters:
                table_docs = _filter_docs_by_month_year(table_docs, month_filters)
            if threshold_query:
                table_docs = _filter_growth_threshold_table_docs(
                    table_docs,
                    question_for_retrieval,
                )

        if years and not month_filters and not any(_year_in_source(doc, years) for doc in docs):
            fallback_k = getattr(retriever, "k", config.RETRIEVAL_K)
            year_docs = _retrieve_year_docs(retriever, question_for_retrieval, years, fallback_k)
            if year_docs:
                merged = []
                seen_ids = set()
                for doc in year_docs + docs:
                    doc_id = doc.metadata.get("chunk_id") or id(doc)
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    merged.append(doc)
                docs = merged

        if guide_query:
            docs = [doc for doc in docs if _is_guide_source(doc)]
            docs = _dedupe_docs(docs)

        if month_filters:
            docs = _filter_docs_by_month_year(docs, month_filters)
            docs = _dedupe_docs(docs)

        if threshold_query:
            docs = _filter_growth_threshold_context_docs(
                docs,
                question_for_retrieval,
                has_threshold_table_docs=bool(table_docs),
            )

        used_web_search = False
        if (
            not docs
            and not table_docs
            and config.WEB_SEARCH_ENABLED
            and not month_filters
            and not guide_query
            and not threshold_query
        ):
            web_results = search_web(question, max_results=config.WEB_SEARCH_MAX_RESULTS)
            docs = web_results_to_documents(web_results)
            used_web_search = bool(docs)

        if (
            knowledge_graph is not None
            and docs
            and not used_web_search
            and not guide_query
            and not threshold_query
        ):
            seed_docs = docs
            docs = knowledge_graph.expand_documents(
                docs,
                question=question,
                max_related_chunks=max_graph_chunks,
            )
            if month_filters:
                filtered_expanded = _filter_docs_by_month_year(docs, month_filters)
                docs = filtered_expanded if filtered_expanded else seed_docs

        if month_filters:
            docs = _dedupe_docs(_filter_docs_by_month_year(docs, month_filters))

        context_str = _format_docs(docs)
        table_context = format_table_documents(table_docs)
        graph_context = (
            "External web search results (no KG context)."
            if used_web_search
            else knowledge_graph.describe_context(question, docs)
            if knowledge_graph is not None
            else "No graph context available."
        )
        answer = answer_chain.invoke(
            {
                "input": question,
                "context": context_str,
                "table_context": table_context,
                "graph_context": graph_context,
                "chat_history": chat_history,
            }
        )
        return {
            "answer": answer,
            "context": docs,
            "table_docs": table_docs,
            "table_context": table_context,
            "graph_context": graph_context,
        }

    rag_chain = RunnableLambda(full_rag)

    logger.info("RAG chain assembled successfully.")
    return rag_chain


def format_source_documents(documents: list[Document]) -> str:
    """Format retrieved source documents for display.

    Args:
        documents: List of retrieved Document objects.

    Returns:
        A human-readable string listing each source and a snippet
        of its content.
    """
    if not documents:
        return "No source documents retrieved."

    lines: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:200].replace("\n", " ")
        lines.append(f"[{i}] {source}\n    {snippet}…")

    return "\n\n".join(lines)
