"""MarineRAG-DSS – Streamlit web application.

Provides a conversational chatbot interface that uses a RAG pipeline
to answer questions about marine resource management, drawing from
environmental data, marine laws, and fisheries knowledge.
"""

from __future__ import annotations

import logging
import os

import streamlit as st

from src.chatbot import MarineChatbot
from src.config import LLMMode, config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "llm_mode" not in st.session_state:
        st.session_state.llm_mode = LLMMode.API
    if "setup_done" not in st.session_state:
        st.session_state.setup_done = False
    if "setup_error" not in st.session_state:
        st.session_state.setup_error = None


def _setup_chatbot(llm_mode: str, force_rebuild: bool = False) -> None:
    """Instantiate and set up the MarineChatbot; store in session state."""
    with st.spinner("Loading knowledge base and preparing RAG pipeline…"):
        try:
            bot = MarineChatbot(
                llm_mode=llm_mode,
                force_rebuild=force_rebuild,
            )
            bot.setup()
            st.session_state.chatbot = bot
            st.session_state.llm_mode = llm_mode
            st.session_state.setup_done = True
            st.session_state.setup_error = None
            logger.info("Chatbot setup complete (mode=%s).", llm_mode)
        except Exception as exc:
            logger.exception("Chatbot setup failed.")
            st.session_state.setup_done = False
            st.session_state.setup_error = str(exc)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("### LLM Mode")
    llm_mode = st.sidebar.radio(
        "Select LLM backend:",
        options=[LLMMode.API, LLMMode.LOCAL],
        format_func=lambda m: "☁️ OpenAI API" if m == LLMMode.API else "💻 Local (Ollama)",
        index=0 if st.session_state.llm_mode == LLMMode.API else 1,
        key="llm_mode_radio",
    )

    if llm_mode == LLMMode.API:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Required for API mode. Set via OPENAI_API_KEY env variable or enter here.",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.sidebar.markdown(
            f"**Model:** `{config.OPENAI_MODEL}`  \n"
            f"**Embeddings:** `{config.OPENAI_EMBEDDING_MODEL}`"
        )
    else:
        st.sidebar.markdown(
            f"**Ollama URL:** `{config.OLLAMA_BASE_URL}`  \n"
            f"**Model:** `{config.OLLAMA_MODEL}`  \n"
            f"**Embeddings:** `{config.OLLAMA_EMBEDDING_MODEL}`  \n\n"
            "_Ensure Ollama is running locally with the required models._"
        )

    st.sidebar.divider()

    st.sidebar.markdown("### Knowledge Base")
    force_rebuild = st.sidebar.checkbox(
        "Rebuild vector index",
        value=False,
        help="Force re-indexing of all documents (slower; use after adding new data).",
    )

    if st.sidebar.button("🚀 Initialise / Reinitialise", use_container_width=True):
        _setup_chatbot(llm_mode, force_rebuild=force_rebuild)

    if st.session_state.setup_done:
        st.sidebar.success("✅ Chatbot ready")
    elif st.session_state.setup_error:
        st.sidebar.error(f"❌ Setup failed:\n{st.session_state.setup_error}")
    else:
        st.sidebar.info("Click **Initialise** to start.")

    st.sidebar.divider()
    st.sidebar.markdown("### Session")
    if st.sidebar.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.reset_history()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Data sources:**\n"
        "- 🌊 Environmental data\n"
        "- ⚖️ Marine laws & regulations\n"
        "- 🐟 Fisheries & aquaculture data\n\n"
        "*Add more `.txt` or `.pdf` files to `data/sample_docs/` "
        "and reinitialise.*"
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def _render_header() -> None:
    col1, col2 = st.columns([1, 6])
    with col1:
        st.markdown("# 🌊")
    with col2:
        st.title(config.APP_TITLE)
        st.caption(
            "A Retrieval-Augmented Generation (RAG) Decision Support System "
            "for marine resource management — powered by multi-source environmental, "
            "legal, and fisheries knowledge."
        )
    st.divider()


def _render_chat() -> None:
    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Source documents"):
                    st.text(msg["sources"])

    # Chat input
    if prompt := st.chat_input(
        "Ask about marine regulations, fish stocks, environmental data…",
        disabled=not st.session_state.setup_done,
    ):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                try:
                    result = st.session_state.chatbot.chat(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as exc:
                    logger.exception("Error during chat.")
                    answer = f"⚠️ An error occurred: {exc}"
                    sources = ""

            st.markdown(answer)
            if sources:
                with st.expander("📄 Source documents"):
                    st.text(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


def _render_example_queries() -> None:
    if st.session_state.setup_done and not st.session_state.messages:
        st.markdown("### 💡 Example questions")
        examples = [
            "What is the minimum conservation reference size for Atlantic Cod?",
            "What are the main objectives of the EU Common Fisheries Policy?",
            "What temperature range is acceptable for healthy marine ecosystems?",
            "Explain the difference between MSY and MEY in fisheries management.",
            "What are the regulations under MARPOL regarding plastic discharge?",
            "What is the 30×30 marine protection target and when must it be achieved?",
            "How are Bluefin tuna stocks currently classified?",
            "What fishing gear is prohibited near deep-sea sensitive areas?",
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(example, use_container_width=True, key=f"ex_{i}"):
                # Trigger a chat with the example question
                st.session_state.messages.append({"role": "user", "content": example})
                with st.spinner("Searching knowledge base…"):
                    try:
                        result = st.session_state.chatbot.chat(example)
                        answer = result["answer"]
                        sources = result["sources"]
                    except Exception as exc:
                        logger.exception("Error during example chat.")
                        answer = f"⚠️ An error occurred: {exc}"
                        sources = ""
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()
    _render_sidebar()
    _render_header()

    if not st.session_state.setup_done:
        st.info(
            "👈 Select an LLM mode in the sidebar and click **Initialise** to start."
        )
        return

    _render_example_queries()
    _render_chat()


if __name__ == "__main__":
    main()
