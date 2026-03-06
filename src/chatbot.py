"""High-level chatbot interface for MarineRAG-DSS.

Provides the ``MarineChatbot`` class that wires together document
loading, vector store management, LLM selection, and the RAG chain
into a single easy-to-use object.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.config import LLMMode, config
from src.document_processor import load_and_split
from src.rag_chain import build_rag_chain, format_source_documents
from src.vector_store import get_or_build_vector_store, get_retriever

logger = logging.getLogger(__name__)


class MarineChatbot:
    """RAG-powered chatbot for marine resource management.

    Supports two operation modes controlled by ``llm_mode``:

    * ``LLMMode.API``   — uses OpenAI's API for both the LLM and embeddings.
    * ``LLMMode.LOCAL`` — uses a locally running Ollama instance.

    Example::

        bot = MarineChatbot(llm_mode=LLMMode.API)
        bot.setup()
        result = bot.chat("What are the fishing restrictions in Zone A?")
        print(result["answer"])
    """

    def __init__(
        self,
        llm_mode: str = LLMMode.API,
        data_dir: Optional[str] = None,
        store_path: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Initialise the chatbot (does not yet load data or build index).

        Args:
            llm_mode: ``"api"`` for OpenAI or ``"local"`` for Ollama.
            data_dir: Override for the source-document directory.
            store_path: Override for the vector-store persistence directory.
            force_rebuild: If True, rebuild the vector store even if one exists.
        """
        self.llm_mode = llm_mode
        self.data_dir = data_dir
        self.store_path = store_path
        self.force_rebuild = force_rebuild

        self._chain = None
        self._chat_history: list = []
        self._ready = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load documents, build/load the vector store, and assemble the RAG chain.

        Must be called before :meth:`chat`.
        """
        logger.info("Setting up MarineChatbot (mode=%s)…", self.llm_mode)

        embeddings, llm = self._get_models()

        chunks = load_and_split(self.data_dir)
        vector_store = get_or_build_vector_store(
            documents=chunks,
            embeddings=embeddings,
            store_path=self.store_path,
            force_rebuild=self.force_rebuild,
        )
        retriever = get_retriever(vector_store)
        self._chain = build_rag_chain(llm, retriever)
        self._ready = True
        logger.info("MarineChatbot is ready.")

    def _get_models(self):
        """Return (embeddings, llm) for the configured mode."""
        if self.llm_mode == LLMMode.LOCAL:
            from src.local_llm_module import get_local_embeddings, get_local_llm

            return get_local_embeddings(), get_local_llm()
        else:
            from src.api_llm_module import get_api_embeddings, get_api_llm

            return get_api_embeddings(), get_api_llm()

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def chat(self, question: str) -> dict:
        """Ask a question and get a grounded answer from the RAG chain.

        Args:
            question: User's natural-language question.

        Returns:
            A dict with keys:
            * ``"answer"``  — the generated answer string.
            * ``"sources"`` — formatted string listing source documents.
            * ``"context"`` — list of raw retrieved Document objects.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if not self._ready or self._chain is None:
            raise RuntimeError(
                "Chatbot is not initialised. Call setup() first."
            )

        logger.debug("User question: %s", question)

        result = self._chain.invoke(
            {
                "input": question,
                "chat_history": self._chat_history,
            }
        )

        answer = result.get("answer", "")
        context_docs = result.get("context", [])

        # Update conversation history
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": format_source_documents(context_docs),
            "context": context_docs,
        }

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self._chat_history = []
        logger.info("Chat history cleared.")

    @property
    def is_ready(self) -> bool:
        """Return True if the chatbot has been set up successfully."""
        return self._ready
