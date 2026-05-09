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

    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    )

    # Ollama local LLM settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )

    # Local embedding provider settings
    LOCAL_EMBEDDINGS_PROVIDER: str = os.getenv("LOCAL_EMBEDDINGS_PROVIDER", "ollama")
    HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "Vietnamese_Embedding_v2")
    HF_EMBEDDING_DEVICE: str = os.getenv("HF_EMBEDDING_DEVICE", "cpu")
    HF_EMBEDDING_TRUST_REMOTE_CODE: bool = os.getenv(
        "HF_EMBEDDING_TRUST_REMOTE_CODE", "false"
    ).lower() in ("1", "true", "yes", "y")

    # Vector store settings
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Knowledge graph settings
    KG_STORE_PATH: str = os.getenv("KG_STORE_PATH", "data/knowledge_graph")
    KG_MAX_EXPANSION_CHUNKS: int = int(os.getenv("KG_MAX_EXPANSION_CHUNKS", "5"))
    KG_MAX_GRAPH_ENTITIES: int = int(os.getenv("KG_MAX_GRAPH_ENTITIES", "8"))
    KG_STORE_BACKEND: str = os.getenv("KG_STORE_BACKEND", "json")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    NEO4J_BATCH_SIZE: int = int(os.getenv("NEO4J_BATCH_SIZE", "500"))
    NEO4J_CLEAR_ON_REBUILD: bool = os.getenv(
        "NEO4J_CLEAR_ON_REBUILD", "true"
    ).lower() in ("1", "true", "yes", "y")

    # Retrieval settings
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
    TABLE_RETRIEVAL_K: int = int(os.getenv("TABLE_RETRIEVAL_K", "5"))
    TABLE_STRUCTURED_MIN_SCORE: int = int(
        os.getenv("TABLE_STRUCTURED_MIN_SCORE", "1")
    )
    TABLE_STORE_BACKEND: str = os.getenv("TABLE_STORE_BACKEND", "mysql")

    # Table MySQL settings
    TABLE_MYSQL_HOST: str = os.getenv("TABLE_MYSQL_HOST", "localhost")
    TABLE_MYSQL_PORT: int = int(os.getenv("TABLE_MYSQL_PORT", "3306"))
    TABLE_MYSQL_DB: str = os.getenv("TABLE_MYSQL_DB", "marine_rag_dss")
    TABLE_MYSQL_USER: str = os.getenv("TABLE_MYSQL_USER", "")
    TABLE_MYSQL_PASSWORD: str = os.getenv("TABLE_MYSQL_PASSWORD", "")
    TABLE_MYSQL_TABLE: str = os.getenv("TABLE_MYSQL_TABLE", "table_rows")
    TABLE_MYSQL_CANDIDATE_LIMIT: int = int(
        os.getenv("TABLE_MYSQL_CANDIDATE_LIMIT", "400")
    )
    TABLE_MYSQL_TOKEN_LIMIT: int = int(os.getenv("TABLE_MYSQL_TOKEN_LIMIT", "8"))

    # Web search fallback settings
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

    # Data directory
    DATA_DIR: str = os.getenv("DATA_DIR", "data/sample_docs")
    GUIDE_VIDEO_LINKS_PATH: str = os.getenv(
        "GUIDE_VIDEO_LINKS_PATH", "data/User_guide/video_links.txt"
    )

    # Application settings
    APP_TITLE: str = os.getenv("APP_TITLE", "MarineRAG DSS")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Auth settings
    DEFAULT_USER_ROLE: str = os.getenv("DEFAULT_USER_ROLE", "citizen")
    OAUTH_REDIRECT_URI: str = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501")
    OAUTH_GOOGLE_CLIENT_ID: str = os.getenv("OAUTH_GOOGLE_CLIENT_ID", "")
    OAUTH_GOOGLE_CLIENT_SECRET: str = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET", "")
    OAUTH_MICROSOFT_CLIENT_ID: str = os.getenv("OAUTH_MICROSOFT_CLIENT_ID", "")
    OAUTH_MICROSOFT_CLIENT_SECRET: str = os.getenv("OAUTH_MICROSOFT_CLIENT_SECRET", "")
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", OAUTH_REDIRECT_URI)
    EMAIL_VERIFY_TOKEN_HOURS: int = int(os.getenv("EMAIL_VERIFY_TOKEN_HOURS", "24"))
    PASSWORD_RESET_TOKEN_HOURS: int = int(os.getenv("PASSWORD_RESET_TOKEN_HOURS", "2"))

    # SMTP settings (optional for email verification/password reset)
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "")
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    # Database settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "marine_rag_dss")

    # RAG prompt template
    RAG_SYSTEM_PROMPT: str = (
        "You are an expert marine resource management assistant. "
        "Use the retrieved context and knowledge-graph hints below to answer the user's question accurately and concisely. "
        "If the context does not contain enough information, say so clearly. "
        "Always cite which document or section your answer comes from when possible.\n\n"
        "Knowledge graph hints:\n{graph_context}\n\n"
        "Structured table data:\n{table_context}\n\n"
        "Context:\n{context}"
    )


config = Config()
