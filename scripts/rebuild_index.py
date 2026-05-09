import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import config
from src.document_processor import load_and_split, load_table_documents
from src.kg_store import get_or_build_knowledge_graph
from src.local_llm_module import get_local_embeddings
from src.table_store import MySQLTableStore
from src.vector_store import build_vector_store


def _store_size(store) -> int:
    index = getattr(store, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return int(index.ntotal)
    docstore = getattr(store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        return len(docstore._dict)
    return 0


def main() -> None:
    chunks = load_and_split(include_spreadsheets=False)
    print("data_dir=", config.DATA_DIR)
    print("chunks=", len(chunks))
    table_docs = load_table_documents()
    print("table_rows=", len(table_docs))

    embeddings = get_local_embeddings()
    vector_store = build_vector_store(
        chunks,
        embeddings,
        store_path=config.VECTOR_STORE_PATH,
    )
    print("vector_store_docs=", _store_size(vector_store))

    table_backend = (config.TABLE_STORE_BACKEND or "mysql").strip().lower()
    if table_backend == "mysql":
        table_store = MySQLTableStore.from_config()
        table_store.ensure_schema()
        if table_docs:
            table_store.ingest_documents(table_docs, force_rebuild=True)
            print("table_store_rows=", len(table_docs))
        else:
            print("table_store_rows=0")
    else:
        print("table_store_rows=0")

    knowledge_graph = get_or_build_knowledge_graph(
        documents=chunks,
        store_path=config.KG_STORE_PATH,
        force_rebuild=True,
    )
    chunk_map = getattr(knowledge_graph, "chunk_to_entities", None)
    if isinstance(chunk_map, dict):
        kg_chunks = len(chunk_map)
    else:
        chunk_count = getattr(knowledge_graph, "chunk_count", None)
        kg_chunks = chunk_count() if callable(chunk_count) else 0
    print("kg_chunks=", kg_chunks)


if __name__ == "__main__":
    main()
