"""RAG chain construction for MarineRAG-DSS.

Builds a conversational retrieval-augmented generation (RAG) chain
that combines a vector-store retriever with a chat LLM to answer
questions about marine resource management.

The chain is built using LangChain Expression Language (LCEL):
  input → retrieve context → format prompt → LLM → parse output
"""

from __future__ import annotations

import logging


from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import config

logger = logging.getLogger(__name__)


def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page contents for use in the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    system_prompt: str | None = None,
) -> Runnable:
    """Assemble a RAG chain from an LLM and a retriever using LCEL.

    The chain follows a *retrieve-then-read* pattern:
    1. The retriever fetches the ``k`` most relevant chunks for the query.
    2. The LLM generates an answer conditioned on those chunks.

    Args:
        llm: Chat language model (OpenAI or Ollama).
        retriever: Document retriever backed by the vector store.
        system_prompt: Optional override for the system/context prompt.
                       Must contain the ``{context}`` placeholder.

    Returns:
        A LangChain ``Runnable`` that accepts
        ``{"input": <question>, "chat_history": [...]}`` (chat_history optional)
        and returns ``{"answer": <response>, "context": [<docs>]}``.
    """
    system_prompt = system_prompt or config.RAG_SYSTEM_PROMPT

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
        docs = retriever.invoke(question)
        context_str = _format_docs(docs)
        answer = answer_chain.invoke(
            {"input": question, "context": context_str, "chat_history": chat_history}
        )
        return {"answer": answer, "context": docs}

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

    lines: List[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:200].replace("\n", " ")
        lines.append(f"[{i}] {source}\n    {snippet}…")

    return "\n\n".join(lines)
