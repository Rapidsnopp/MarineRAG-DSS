"""Tests for the chatbot module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import LLMMode


class TestMarineChatbot:
    def test_init_defaults(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot()
        assert bot.llm_mode == LLMMode.API
        assert not bot.is_ready

    def test_chat_raises_before_setup(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot()
        try:
            bot.chat("What is the MSY for Atlantic Cod?")
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "not initialised" in str(exc)

        assert raised

    def test_setup_api_mode(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot(llm_mode=LLMMode.API)

        mock_embeddings = MagicMock()
        mock_llm = MagicMock()
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_chain = MagicMock()
        mock_graph = MagicMock()

        with patch("src.chatbot.load_and_split", return_value=[MagicMock()]), \
            patch("src.chatbot.config.TABLE_STORE_BACKEND", "memory"), \
            patch("src.chatbot.load_table_documents", return_value=[]), \
            patch("src.chatbot.get_or_build_vector_store", return_value=mock_vector_store), \
            patch("src.chatbot.get_retriever", return_value=mock_retriever), \
            patch("src.chatbot.get_or_build_knowledge_graph", return_value=mock_graph), \
            patch("src.chatbot.build_rag_chain", return_value=mock_chain), \
            patch.object(MarineChatbot, "_get_models", return_value=(mock_embeddings, mock_llm)):
            bot.setup()

        assert bot.is_ready

    def test_setup_local_mode(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot(llm_mode=LLMMode.LOCAL)

        mock_embeddings = MagicMock()
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_graph = MagicMock()

        with patch("src.chatbot.load_and_split", return_value=[MagicMock()]), \
            patch("src.chatbot.config.TABLE_STORE_BACKEND", "memory"), \
            patch("src.chatbot.load_table_documents", return_value=[]), \
            patch("src.chatbot.get_or_build_vector_store", return_value=MagicMock()), \
            patch("src.chatbot.get_retriever", return_value=MagicMock()), \
            patch("src.chatbot.get_or_build_knowledge_graph", return_value=mock_graph), \
            patch("src.chatbot.build_rag_chain", return_value=mock_chain), \
            patch.object(MarineChatbot, "_get_models", return_value=(mock_embeddings, mock_llm)):
            bot.setup()

        assert bot.is_ready

    def test_chat_returns_expected_keys(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "Atlantic cod MCRS is 35 cm.",
            "context": [],
            "graph_context": "",
        }
        bot._chain = mock_chain
        bot._ready = True

        result = bot.chat("What is the minimum size for Atlantic cod?")

        assert "answer" in result
        assert "sources" in result
        assert "context" in result
        assert "table_docs" in result
        assert result["answer"] == "Atlantic cod MCRS is 35 cm."

    def test_chat_updates_history(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Test answer.", "context": [], "graph_context": ""}
        bot._chain = mock_chain
        bot._ready = True

        bot.chat("First question")
        bot.chat("Second question")

        assert len(bot._chat_history) == 4  # 2 human + 2 AI messages

    def test_reset_history(self) -> None:
        from src.chatbot import MarineChatbot

        bot = MarineChatbot()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"answer": "Answer.", "context": [], "graph_context": ""}
        bot._chain = mock_chain
        bot._ready = True

        bot.chat("A question")
        assert len(bot._chat_history) == 2

        bot.reset_history()
        assert len(bot._chat_history) == 0
