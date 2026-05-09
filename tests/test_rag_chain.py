"""Tests for the rag_chain module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# format_source_documents
# ---------------------------------------------------------------------------


class TestFormatSourceDocuments:
    def test_formats_documents(self) -> None:
        from src.rag_chain import format_source_documents

        docs = [
            Document(
                page_content="Atlantic Cod minimum size is 35 cm.",
                metadata={"source": "marine_laws.txt"},
            ),
            Document(
                page_content="Ocean pH is 8.1.",
                metadata={"source": "environmental_data.txt"},
            ),
        ]
        result = format_source_documents(docs)
        assert "marine_laws.txt" in result
        assert "environmental_data.txt" in result
        assert "[1]" in result
        assert "[2]" in result

    def test_returns_message_for_empty_list(self) -> None:
        from src.rag_chain import format_source_documents

        result = format_source_documents([])
        assert "No source documents" in result

    def test_handles_missing_source_metadata(self) -> None:
        from src.rag_chain import format_source_documents

        docs = [Document(page_content="Some content.", metadata={})]
        result = format_source_documents(docs)
        assert "Unknown" in result

    def test_truncates_long_content(self) -> None:
        from src.rag_chain import format_source_documents

        long_text = "A" * 500
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        result = format_source_documents(docs)
        # Snippet should be <= 200 chars plus ellipsis
        assert len(result) < len(long_text)


# ---------------------------------------------------------------------------
# build_rag_chain
# ---------------------------------------------------------------------------


class TestBuildRagChain:
    def test_returns_runnable(self) -> None:
        from langchain_core.runnables import Runnable

        from src.rag_chain import build_rag_chain

        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        chain = build_rag_chain(mock_llm, mock_retriever)
        assert chain is not None
        assert isinstance(chain, Runnable)

    def test_invokes_retriever_on_question(self) -> None:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from src.rag_chain import build_rag_chain

        fake_llm = FakeListChatModel(responses=["Test answer about cod size."])
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Cod MCRS is 35 cm.", metadata={"source": "laws.txt"})
        ]

        chain = build_rag_chain(fake_llm, mock_retriever)
        result = chain.invoke({"input": "What is the cod size limit?"})

        mock_retriever.invoke.assert_called_once_with("What is the cod size limit?")
        assert "answer" in result
        assert "context" in result
        assert isinstance(result["answer"], str)

    def test_uses_custom_system_prompt(self) -> None:
        from src.rag_chain import build_rag_chain

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer.")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        custom_prompt = "You are a helpful assistant. Context: {context}"

        with patch("src.rag_chain.ChatPromptTemplate") as mock_template:
            mock_template.from_messages.return_value = MagicMock()
            build_rag_chain(mock_llm, mock_retriever, system_prompt=custom_prompt)
            call_args = mock_template.from_messages.call_args[0][0]
            assert any(custom_prompt in str(msg) for msg in call_args)

    def test_threshold_query_drops_irrelevant_vector_context(self) -> None:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from src.rag_chain import build_rag_chain

        fake_llm = FakeListChatModel(responses=["Threshold answer."])
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(
                page_content="MARPOL regulations for marine pollution.",
                metadata={"source": "data/Report_Thai_Binh_2024/marpol_report.pdf"},
            )
        ]

        table_store = MagicMock()
        table_store.query.return_value = [
            Document(
                page_content="Thông số: pH | Rất phù hợp: 6,2 - 7,7",
                metadata={
                    "source": "data/Growth_threshold/1. Chuyên gia thống nhất_CÂY BẦN CHUA.xlsx",
                    "sheet": "3 Ngưỡng",
                    "row_index": 7,
                },
            )
        ]

        chain = build_rag_chain(fake_llm, mock_retriever, table_store=table_store)
        result = chain.invoke(
            {"input": "Cây bần chua có ngưỡng sinh trưởng phù hợp với các thông số nào?"}
        )

        table_store.query.assert_called_once()
        assert result["context"] == []
        assert len(result["table_docs"]) == 1
        source = str(result["table_docs"][0].metadata.get("source", "")).lower()
        assert "ban chua" in source or "bần chua" in source

    def test_threshold_query_filters_table_docs_by_species(self) -> None:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from src.rag_chain import build_rag_chain

        fake_llm = FakeListChatModel(responses=["Threshold answer."])
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        table_store = MagicMock()
        table_store.query.return_value = [
            Document(
                page_content="Thông số: pH | Rất phù hợp: 6,2 - 7,7",
                metadata={
                    "source": "data/Growth_threshold/1. Chuyên gia thống nhất_CÂY BẦN CHUA.xlsx",
                    "sheet": "3 Ngưỡng",
                    "row_index": 7,
                },
            ),
            Document(
                page_content="Thông số: Độ mặn | Rất phù hợp: 15 - 25",
                metadata={
                    "source": "data/Growth_threshold/2. Chuyên gia thống nhất_HÀU.xlsx",
                    "sheet": "3 Ngưỡng",
                    "row_index": 8,
                },
            ),
        ]

        chain = build_rag_chain(fake_llm, mock_retriever, table_store=table_store)
        result = chain.invoke(
            {"input": "Ngưỡng sinh trưởng phù hợp của cây bần chua gồm thông số nào?"}
        )

        sources = [
            str(doc.metadata.get("source", "")).lower()
            for doc in result["table_docs"]
        ]
        assert sources
        assert any("ban chua" in source or "bần chua" in source for source in sources)
        assert all("hàu" not in source and "hau" not in source for source in sources)
