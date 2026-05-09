import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api_llm_module import get_api_llm
from src.config import config
from src.document_processor import load_and_split, load_table_documents
from src.kg_store import get_or_build_knowledge_graph
from src.local_llm_module import get_local_embeddings
from src.rag_chain import build_rag_chain, format_source_documents
from src.table_store import MySQLTableStore, TableStore
from src.vector_store import build_vector_store, get_retriever


def main() -> None:
    question = "What does MARPOL regulate regarding marine pollution?"

    chunks = load_and_split(include_spreadsheets=False)
    print("chunks=", len(chunks))
    table_docs = load_table_documents()
    print("table_rows=", len(table_docs))

    embeddings = get_local_embeddings()
    vector_store = build_vector_store(chunks, embeddings, store_path=config.VECTOR_STORE_PATH)
    retriever = get_retriever(vector_store)

    table_store = None
    table_backend = (config.TABLE_STORE_BACKEND or "mysql").strip().lower()
    if table_backend == "mysql":
        try:
            table_store = MySQLTableStore.from_config()
            table_store.ensure_schema()
            if table_docs:
                table_store.ingest_documents(table_docs, force_rebuild=True)
        except Exception:
            table_store = TableStore(table_docs) if table_docs else None
    else:
        table_store = TableStore(table_docs) if table_docs else None

    knowledge_graph = get_or_build_knowledge_graph(
        documents=chunks,
        store_path=config.KG_STORE_PATH,
        force_rebuild=True,
    )

    llm = get_api_llm()
    rag_chain = build_rag_chain(
        llm,
        retriever,
        knowledge_graph=knowledge_graph,
        table_store=table_store,
    )

    result = rag_chain.invoke({"input": question, "chat_history": []})
    print("Q:", question)
    print("A:", result.get("answer", ""))
    combined_docs = (result.get("context", []) or []) + (result.get("table_docs", []) or [])
    print("Sources:\n", format_source_documents(combined_docs))


if __name__ == "__main__":
    main()
