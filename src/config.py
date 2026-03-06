"""Configuration management for MarineRAG-DSS."""

from __future__ import annotations

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class LLMMode(str, Enum):
    """LLM operation mode."""

    API = "api"
    LOCAL = "local"


class Config:
    """Central configuration for the MarineRAG-DSS application."""

    # LLM mode
    LLM_MODE: str = os.getenv("LLM_MODE", LLMMode.API)

    # OpenAI API settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Ollama local LLM settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )

    # Vector store settings
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval settings
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))

    # Data directory
    DATA_DIR: str = os.getenv("DATA_DIR", "data/sample_docs")

    # Application settings
    APP_TITLE: str = os.getenv("APP_TITLE", "MarineRAG DSS")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # RAG prompt template
    RAG_SYSTEM_PROMPT: str = (
        "You are an expert marine resource management assistant. "
        "Use the retrieved context below to answer the user's question accurately and concisely. "
        "If the context does not contain enough information, say so clearly. "
        "Always cite which document or section your answer comes from when possible.\n\n"
        "Context:\n{context}"
    )


config = Config()
