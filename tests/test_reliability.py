"""Phase 2: no-answer citations, health probes, and indexed-only stats.

These tests use temporary directories and mocks. They do not touch the
project's data/ or vector_db/ directories.
"""

from __future__ import annotations

import urllib.request

from langchain_core.documents import Document

from backend.context_builder import build_context, snippet_text
from backend.conversation import ConversationManager
from backend.hybrid_retriever import HybridRetriever
from backend.llm import NO_ANSWER
from backend.main import check_ollama
from backend.rag_pipeline import RAGPipeline
from backend.vector_store import vector_store_reachable
from tests.conftest import LANGCHAIN_PDF


class _Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_check_ollama_success_uses_the_given_timeout(monkeypatch):
    seen = {}

    def urlopen(request, timeout=None):
        seen["timeout"] = timeout
        seen["url"] = request.full_url
        return _Response(200)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert check_ollama("http://127.0.0.1:11434", timeout=0.25) is True
    assert seen["timeout"] == 0.25
    assert seen["url"] == "http://127.0.0.1:11434/api/version"


def test_check_ollama_timeout_is_a_model_outage(monkeypatch):
    def urlopen(_request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert check_ollama("http://127.0.0.1:11434", timeout=0.1) is False


def test_check_ollama_rejects_a_non_positive_timeout(monkeypatch):
    seen = {}

    def urlopen(_request, timeout=None):
        seen["timeout"] = timeout
        return _Response(200)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert check_ollama("http://127.0.0.1:11434/", timeout=0) is True
    assert seen["timeout"] == 1.0


def test_vector_store_reachable_reads_the_collection_count():
    class Collection:
        def count(self):
            return 3

    class Store:
        _collection = Collection()

    assert vector_store_reachable(Store()) is True


def test_vector_store_error_is_reported_as_unreachable():
    class Collection:
        def count(self):
            raise RuntimeError("database is locked")

    class Store:
        _collection = Collection()

    assert vector_store_reachable(Store()) is False
    assert vector_store_reachable(object()) is False


def _pipeline(tmp_path, monkeypatch, answer, citations):
    document = Document(
        page_content="A retriever returns relevant documents.",
        metadata={"source": "notes.pdf", "page": 2, "chunk_id": 0},
    )

    monkeypatch.setattr("backend.rag_pipeline.rewrite_query", lambda **_kwargs: "standalone")
    monkeypatch.setattr(
        "backend.rag_pipeline.select_mmr_documents",
        lambda **_kwargs: [(document, 1.0)],
    )
    monkeypatch.setattr(
        "backend.rag_pipeline.rerank_documents",
        lambda **_kwargs: [document],
    )
    monkeypatch.setattr(
        "backend.rag_pipeline.build_context",
        lambda **_kwargs: ("context", citations),
    )
    monkeypatch.setattr(
        "backend.rag_pipeline.generate_answer",
        lambda **_kwargs: answer,
    )

    class Hybrid:
        def search(self, **_kwargs):
            return [(document, 1.0)]

    pipeline = RAGPipeline(
        embedding_model=object(),
        vector_store=object(),
        bm25_retriever=None,
        reranker=object(),
        llm=object(),
        conversation_manager=ConversationManager(db_path=tmp_path / "conversations.db"),
    )
    pipeline.hybrid_retriever = Hybrid()
    return pipeline


def test_no_answer_omits_citations_in_the_response_and_the_history(tmp_path, monkeypatch):
    citations = [{"reference": "[Source 1]", "source": "notes.pdf", "page": 2, "chunk_id": 0}]
    pipeline = _pipeline(tmp_path, monkeypatch, NO_ANSWER, citations)
    labels = {"notes.pdf": {"document_id": "doc-1", "original_filename": "Notes.pdf"}}

    result = pipeline.ask(
        "What is the capital of France?",
        "session-1",
        citation_labels=labels,
    )

    assert result["answer"] == NO_ANSWER
    assert result["citations"] == []

    stored = pipeline.conversation_manager.get_session("session-1")
    assistant = [message for message in stored["messages"] if message["role"] == "assistant"]
    assert assistant[0]["metadata"]["citations"] == []


def test_grounded_answer_keeps_citations(tmp_path, monkeypatch):
    citations = [{"reference": "[Source 1]", "source": "notes.pdf", "page": 2, "chunk_id": 0}]
    pipeline = _pipeline(tmp_path, monkeypatch, "A retriever returns documents. [Source 1]", citations)

    result = pipeline.ask("What does a retriever do?", "session-1")

    assert result["citations"] == citations
    stored = pipeline.conversation_manager.get_session("session-1")
    assistant = [message for message in stored["messages"] if message["role"] == "assistant"]
    assert assistant[0]["metadata"]["citations"] == citations


def test_context_snippet_is_a_short_excerpt():
    long_text = "retriever " * 80
    document = Document(
        page_content=long_text,
        metadata={"source": "notes.pdf", "page": 3, "chunk_id": 1},
    )

    _context, citations = build_context([document])

    assert len(citations) == 1
    assert citations[0]["snippet"] == snippet_text(long_text)
    assert len(citations[0]["snippet"]) <= 401
    assert citations[0]["source"] == "notes.pdf"
    assert citations[0]["page"] == 3


def test_hybrid_search_applies_one_filter_to_dense_and_bm25(monkeypatch):
    seen = []
    filters = {"source": {"$in": ["a.pdf", "b.pdf"]}}

    def dense(**kwargs):
        seen.append(("dense", kwargs["filters"]))
        return []

    class Sparse:
        def search(self, query, top_k, filters=None):
            seen.append(("bm25", filters))
            return []

    monkeypatch.setattr("backend.hybrid_retriever.search_documents", dense)
    HybridRetriever(object(), Sparse()).search("retriever", top_k=1, fetch_k=2, filters=filters)

    assert seen == [("dense", filters), ("bm25", filters)]


def test_pipeline_drops_chunks_outside_the_selected_documents(tmp_path, monkeypatch):
    inside = Document(page_content="inside the selection", metadata={"source": "keep.pdf", "page": 1})
    outside = Document(page_content="outside the selection", metadata={"source": "drop.pdf", "page": 4})
    seen = {}

    monkeypatch.setattr("backend.rag_pipeline.rewrite_query", lambda **_kwargs: "question")
    monkeypatch.setattr(
        "backend.rag_pipeline.select_mmr_documents",
        lambda **kwargs: kwargs["results"],
    )
    monkeypatch.setattr(
        "backend.rag_pipeline.rerank_documents",
        lambda **kwargs: kwargs["documents"],
    )

    def capture_context(documents, max_context_chars=12000):
        seen["sources"] = [document.metadata["source"] for document in documents]
        return build_context(documents, max_context_chars)

    monkeypatch.setattr("backend.rag_pipeline.build_context", capture_context)
    monkeypatch.setattr(
        "backend.rag_pipeline.generate_answer",
        lambda **_kwargs: "The selected document says so.",
    )

    class Hybrid:
        def search(self, **_kwargs):
            return [(outside, 0.9), (inside, 0.2)]

    pipeline = RAGPipeline(
        embedding_model=object(),
        vector_store=object(),
        bm25_retriever=None,
        reranker=object(),
        llm=object(),
        conversation_manager=ConversationManager(db_path=tmp_path / "conversations.db"),
    )
    pipeline.hybrid_retriever = Hybrid()

    result = pipeline.ask(
        "What does it say?",
        "session-1",
        filters={"source": "keep.pdf"},
        citation_labels={"keep.pdf": {"document_id": "doc-keep", "original_filename": "Keep.pdf"}},
    )

    assert seen["sources"] == ["keep.pdf"]
    assert result["citations"] == [
        {
            "reference": "[Source 1]",
            "source": "keep.pdf",
            "page": 1,
            "chunk_id": None,
            "snippet": "inside the selection",
            "document_id": "doc-keep",
            "original_filename": "Keep.pdf",
        }
    ]


def test_page_and_chunk_totals_ignore_documents_that_are_not_indexed(service):
    record = service.upload("notes.pdf", LANGCHAIN_PDF)
    assert service.stats()["total_pages"] == record["page_count"]

    service.registry.update_status(
        record["document_id"], status="failed", error_message="stopped"
    )

    stats = service.stats()
    assert stats["total_documents"] == 1
    assert stats["indexed_documents"] == 0
    assert stats["total_pages"] == 0
    assert stats["total_chunks"] == 0
