
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document

if TYPE_CHECKING:  # only needed for type hints; keeps imports light
    from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "pdf_documents"


def create_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Create or connect to the persistent ChromaDB collection."""

    directory = Path(persist_directory) if persist_directory else VECTOR_DB_PATH
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


def vector_store_reachable(vector_store) -> bool:
    """Return whether Chroma answers a cheap read. Never raises."""

    try:
        collection = getattr(vector_store, "_collection", None)
        if collection is None:
            return False
        collection.count()
        return True
    except Exception:
        logger.debug("Vector database is not reachable", exc_info=True)
        return False


def _get_document_ids(chunks: list[Document]) -> list[str]:
    """Generate deterministic IDs for document chunks."""

    ids = []

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", index)

        ids.append(f"{source}_{chunk_id}")

    return ids


def add_documents(vector_store: Chroma, chunks: list[Document]) -> int:
    """
    Add only chunks that do not already exist.

    Existing chunks are never deleted or overwritten.
    Returns the number of newly added chunks.
    """

    if not chunks:
        return 0

    ids = _get_document_ids(chunks)

    existing_records = vector_store._collection.get(
        ids=ids,
        include=["metadatas"],
    )

    existing_ids = set(existing_records["ids"])

    missing_chunks = []
    missing_ids = []

    for chunk, document_id in zip(chunks, ids):
        if document_id not in existing_ids:
            missing_chunks.append(chunk)
            missing_ids.append(document_id)

    if missing_chunks:
        vector_store.add_documents(
            documents=missing_chunks,
            ids=missing_ids,
        )

    return len(missing_chunks)


def replace_source_documents(
    vector_store: Chroma,
    source: str,
    chunks: list[Document],
) -> int:
    """
    Explicitly replace all indexed chunks belonging to one source.

    Use this only when intentionally re-indexing a changed PDF.
    """

    if not chunks:
        raise ValueError("Cannot replace a source with empty chunks.")

    for chunk in chunks:
        if chunk.metadata.get("source") != source:
            raise ValueError(
                "All chunks must belong to the specified source."
            )

    vector_store._collection.delete(
        where={"source": source}
    )

    ids = _get_document_ids(chunks)

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)



def search_documents(
    vector_store: Chroma,
    query: str,
    k: int = 5,
    filters: dict | None = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents using ChromaDB similarity search.

    Args:
        vector_store: Persistent ChromaDB instance.
        query: Search query.
        k: Maximum number of results.
        filters: Optional ChromaDB metadata filters.

    Returns:
        List of (Document, distance) pairs.
        Lower distance indicates greater similarity.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=filters,
    )

    return results


def count_chunks_by_source(vector_store: Chroma) -> dict[str, int]:
    """Return {source: number_of_vectors} for everything stored in ChromaDB.

    Read-only. Used to reconcile the vector store with the document registry.
    """

    records = vector_store._collection.get(include=["metadatas"])
    counts: dict[str, int] = {}

    for metadata in records["metadatas"] or []:
        source = (metadata or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    return counts


def delete_source(vector_store: Chroma, source: str) -> int:
    """Delete every vector belonging to one source. Returns how many were removed."""

    if not source:
        raise ValueError("source cannot be empty.")

    existing = vector_store._collection.get(where={"source": source}, include=[])
    count = len(existing["ids"])

    if count:
        vector_store._collection.delete(where={"source": source})

    return count
