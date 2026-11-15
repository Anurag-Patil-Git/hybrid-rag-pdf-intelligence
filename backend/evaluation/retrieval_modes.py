"""Four retrieval configurations that reuse existing backend components.

Modes
-----
* ``dense`` — Chroma similarity search only
* ``bm25`` — BM25 only
* ``hybrid`` — dense + BM25 fused with Reciprocal Rank Fusion
* ``hybrid_rerank`` — hybrid candidates, then cross-encoder rerank

These paths intentionally skip conversation rewrite, MMR, and LLM answer
generation so retrieval can be scored on its own.
"""

from __future__ import annotations

import time
from typing import Callable

from langchain_core.documents import Document

from backend.bm25_retriever import BM25Retriever
from backend.hybrid_retriever import HybridRetriever
from backend.reranker import rerank_documents
from backend.vector_store import search_documents

RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class RetrievalBundle:
    """Holds one set of models/indexes shared across every evaluation query."""

    def __init__(
        self,
        vector_store,
        bm25_retriever: BM25Retriever | None,
        reranker=None,
        embedding_model=None,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.embedding_model = embedding_model
        self.hybrid = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            rrf_k=60,
        )


def retrieve(
    bundle: RetrievalBundle,
    mode: str,
    query: str,
    *,
    top_k: int,
    fetch_k: int | None = None,
    filters: dict | None = None,
) -> tuple[list[Document], float]:
    """Run one retrieval mode. Returns ``(documents, latency_seconds)``."""

    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    candidate_k = fetch_k or max(top_k * 3, top_k)
    if candidate_k < top_k:
        raise ValueError("fetch_k cannot be smaller than top_k.")

    started = time.perf_counter()

    if mode == "dense":
        pairs = search_documents(
            vector_store=bundle.vector_store,
            query=query,
            k=top_k,
            filters=filters,
        )
        documents = [document for document, _score in pairs]

    elif mode == "bm25":
        if bundle.bm25_retriever is None:
            documents = []
        else:
            pairs = bundle.bm25_retriever.search(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            documents = [document for document, _score in pairs]

    elif mode == "hybrid":
        pairs = bundle.hybrid.search(
            query=query,
            top_k=top_k,
            fetch_k=candidate_k,
            filters=filters,
        )
        documents = [document for document, _score in pairs]

    else:  # hybrid_rerank
        pairs = bundle.hybrid.search(
            query=query,
            top_k=candidate_k,
            fetch_k=candidate_k,
            filters=filters,
        )
        candidates = [document for document, _score in pairs]
        if bundle.reranker is None:
            documents = candidates[:top_k]
        else:
            documents = rerank_documents(
                query=query,
                documents=candidates,
                reranker=bundle.reranker,
                top_n=top_k,
            )

    latency = time.perf_counter() - started
    return documents, latency


def serialize_documents(documents: list[Document]) -> list[dict]:
    return [
        {
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page"),
            "chunk_id": document.metadata.get("chunk_id"),
            "text": document.page_content,
        }
        for document in documents
    ]
