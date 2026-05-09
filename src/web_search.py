"""Web search utilities for fallback retrieval."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    if not query:
        return []
    if DDGS is None:
        logger.warning("duckduckgo_search is not installed; web search is disabled.")
        return []

    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            title = item.get("title") or ""
            url = item.get("href") or ""
            snippet = item.get("body") or ""
            if not (title or url or snippet):
                continue
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def web_results_to_documents(results: list[dict[str, str]]) -> list[Document]:
    documents: list[Document] = []
    for item in results:
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        if not (title or url or snippet):
            continue
        content_parts = [part for part in (title, snippet) if part]
        content = "\n".join(content_parts).strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": url or title or "web",
                    "url": url,
                    "title": title,
                },
            )
        )
    return documents
