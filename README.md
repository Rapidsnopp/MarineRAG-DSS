# 🌊 MarineRAG-DSS

A **Retrieval-Augmented Generation (RAG) based Decision Support System** for marine resource management. The project integrates multi-source environmental, legal, and fisheries data to build an intelligent chatbot that supports marine knowledge retrieval and decision making.

The system includes **two interchangeable LLM modules** for comparison:
- **API-based LLM** — powered by OpenAI (GPT-4o-mini, text-embedding-3-small)
- **Local LLM** — powered by [Ollama](https://ollama.com/) (Llama 3.2, nomic-embed-text)

---

## 📋 Features

- **Multi-source knowledge base**: Integrates environmental data, marine laws & regulations, and fisheries/aquaculture data
- **KG2RAG pipeline**: Retrieves relevant chunks, expands them through a lightweight knowledge graph, then generates grounded, cited answers
- **Two LLM modes**: Seamlessly switch between cloud API and fully local inference
- **FAISS vector store**: Fast similarity search with persistent indexing
- **Structured Excel tables**: Spreadsheet rows persisted to MySQL for structured lookup (tables auto-created)
- **Neo4j knowledge graph backend**: Optional graph persistence for KG expansion
- **Conversational memory**: Maintains multi-turn chat history
- **Streamlit web UI**: Intuitive browser-based chatbot interface with example queries
- **Extensible data ingestion**: Drop `.txt`, `.pdf`, or `.xls/.xlsx` files into `data/sample_docs/` and rebuild the index

---

## 🗂️ Project Structure

```
MarineRAG-DSS/
├── app.py                      # Streamlit web application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── data/
│   ├── sample_docs/            # Source documents (txt, pdf)
│   │   ├── environmental_data.txt
│   │   ├── marine_laws.txt
│   │   └── fisheries_data.txt
│   └── vector_store/           # Persisted FAISS index (auto-created)
├── src/
│   ├── config.py               # Centralised configuration
│   ├── document_processor.py   # Document loading and chunking
│   ├── vector_store.py         # FAISS vector store management
│   ├── api_llm_module.py       # OpenAI API-based LLM + embeddings
│   ├── local_llm_module.py     # Ollama local LLM + embeddings
│   ├── rag_chain.py            # LCEL-based RAG chain
│   ├── table_store.py          # Structured table retrieval utilities
│   └── chatbot.py              # High-level MarineChatbot class
└── tests/
    ├── test_document_processor.py
    ├── test_vector_store.py
    ├── test_rag_chain.py
    └── test_chatbot.py
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Rapidsnopp/MarineRAG-DSS.git
cd MarineRAG-DSS
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (for API mode) or
# configure OLLAMA_BASE_URL / OLLAMA_MODEL (for local mode)
```

### 3. Launch the web application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🔧 LLM Modes

### ☁️ API Mode (OpenAI)

Requires an OpenAI API key. Set it via `.env` or enter it in the sidebar.

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 💻 Local Mode (Ollama)

Requires [Ollama](https://ollama.com/) running locally with the required models pulled.

```bash
# Install Ollama, then pull the required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

---

## ⚙️ Configuration

All settings can be overridden via environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `VECTOR_STORE_PATH` | `data/vector_store` | FAISS index directory |
| `TABLE_STORE_BACKEND` | `mysql` | Table store backend (`mysql` or `memory`) |
| `CHUNK_SIZE` | `800` | Document chunk size (chars) |
| `CHUNK_OVERLAP` | `100` | Chunk overlap (chars) |
| `RETRIEVAL_K` | `5` | Number of chunks retrieved per query |
| `TABLE_RETRIEVAL_K` | `5` | Number of table rows retrieved per query |
| `DATA_DIR` | `data/sample_docs` | Source documents directory |
| `KG_STORE_BACKEND` | `json` | Knowledge graph backend (`json` or `neo4j`) |

### Neo4j knowledge graph store

Set these when using `KG_STORE_BACKEND=neo4j`:

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j user |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database |
| `NEO4J_BATCH_SIZE` | `500` | Batch size for KG ingestion |
| `NEO4J_CLEAR_ON_REBUILD` | `true` | Clear graph on rebuild |

### MySQL table store

Set these when using `TABLE_STORE_BACKEND=mysql`:

| Variable | Default | Description |
|---|---|---|
| `TABLE_MYSQL_HOST` | `localhost` | MySQL host |
| `TABLE_MYSQL_PORT` | `3306` | MySQL port |
| `TABLE_MYSQL_DB` | `marine_rag_dss` | Database name |
| `TABLE_MYSQL_USER` | — | Database user |
| `TABLE_MYSQL_PASSWORD` | — | Database password |
| `TABLE_MYSQL_TABLE` | `table_rows` | Table name for spreadsheet rows |

---

## 📚 Adding Custom Documents

1. Copy `.txt` or `.pdf` files into `data/sample_docs/`
2. Open the app, check **Rebuild vector index** in the sidebar
3. Click **Initialise / Reinitialise**

---

## 🐍 Programmatic Usage

```python
from src.chatbot import MarineChatbot
from src.config import LLMMode

# Use OpenAI API
bot = MarineChatbot(llm_mode=LLMMode.API)
bot.setup()

result = bot.chat("What is the minimum conservation reference size for Atlantic Cod?")
print(result["answer"])
print(result["sources"])   # Which documents were retrieved

# Multi-turn conversation
result2 = bot.chat("And what about European Hake?")
print(result2["answer"])

# Switch to local Ollama
local_bot = MarineChatbot(llm_mode=LLMMode.LOCAL)
local_bot.setup()
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🌊 Sample Knowledge Base Topics

| Document | Contents |
|---|---|
| `environmental_data.txt` | Ocean temperature, MPAs, water quality, pollution monitoring, climate projections |
| `marine_laws.txt` | UNCLOS, CBD, IWC, RFMO regulations (ICES, ICCAT, NAFO), EU CFP, MARPOL |
| `fisheries_data.txt` | Global fisheries overview, stock assessment methods, species profiles, gear types, aquaculture, EBFM |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│                 MarineChatbot                     │
│  ┌─────────────────────────────────────────────┐ │
│  │            KG2RAG Chain (LCEL)              │ │
│  │                                             │ │
│  │  Query ──► Retriever ──► KG Expansion       │ │
│  │                              │              │ │
│  │                    Knowledge Graph          │ │
│  │                              │              │ │
│  │              Chat History ───┤              │ │
│  │                              ▼              │ │
│  │                    Prompt Template          │ │
│  │                              │              │ │
│  │                    LLM (API or Local)       │ │
│  │                              │              │ │
│  │                      Answer String          │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ┌──────────────────┐   ┌────────────────────┐   │
│  │   FAISS Vector   │   │  Document Loader   │   │
│  │      Store       │◄──│   (.txt / .pdf)    │   │
│  └──────────────────┘   └────────────────────┘   │
│  ┌──────────────────┐                              │
│  │ Knowledge Graph   │                              │
│  │  (lightweight)    │                              │
│  └──────────────────┘                              │
└──────────────────────────────────────────────────┘
         ▲                          ▲
         │ embeddings               │ chunks
  ┌──────┴──────┐           ┌───────┴──────┐
  │ OpenAI API  │           │  Ollama Local│
  │  Embeddings │    OR     │  Embeddings  │
  └─────────────┘           └─────────────┘
```
