"""Shared fixtures: hermetic settings, tiny generated PDFs, fake embeddings."""

from __future__ import annotations

import textwrap

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from backend.bm25_retriever import BM25Retriever
from backend.core.config import Settings
from backend.document_registry import DocumentRegistry
from backend.context_builder import apply_citation_labels, snippet_text
from backend.hybrid_retriever import HybridRetriever
from backend.services.document_service import DocumentService
from backend.vector_store import create_vector_store


def make_pdf(*pages: str) -> bytes:
    """Build a small text PDF; each argument becomes one page."""

    document = pymupdf.open()

    for text in pages:
        page = document.new_page()
        wrapped = "\n".join(textwrap.wrap(text, 70)) if text else ""
        if wrapped:
            page.insert_text((50, 60), wrapped, fontsize=11)

    data = document.tobytes()
    document.close()

    return data


LANGCHAIN_PDF = make_pdf(
    "LangChain indexes help applications retrieve documents efficiently. "
    "A retriever returns relevant documents for a query. " * 6,
    "Text splitters divide long documents into overlapping chunks so that "
    "embeddings capture local meaning. " * 6,
)

GEMINI_PDF = make_pdf(
    "Non-native prompt engineering risks include ambiguity in translated "
    "instructions and inconsistent tone across languages. " * 6,
)


class FakePipeline:
    """Stands in for RAGPipeline: real Chroma + BM25 + hybrid search, no ML models."""

    def __init__(self, settings: Settings, conversation_manager=None):
        self.embedding_model = DeterministicFakeEmbedding(size=32)
        self.vector_store = create_vector_store(
            self.embedding_model, persist_directory=settings.vector_db_dir
        )
        self.bm25_retriever = None
        self.hybrid_retriever = HybridRetriever(self.vector_store, None)
        self.conversation_manager = conversation_manager
        self.last_filters = None

    def set_bm25_documents(self, chunks) -> None:
        self.bm25_retriever = BM25Retriever(documents=chunks) if chunks else None
        self.hybrid_retriever.bm25_retriever = self.bm25_retriever

    def ask(self, question, session_id, filters=None, citation_labels=None, **_):
        self.last_filters = filters
        results = self.hybrid_retriever.search(
            query=question, top_k=3, fetch_k=5, filters=filters
        )

        if citation_labels is not None:
            results = [
                (document, score)
                for document, score in results
                if document.metadata.get("source") in citation_labels
            ]

        citations = apply_citation_labels(
            [
                {
                    "source": document.metadata["source"],
                    "page": document.metadata["page"],
                    "chunk_id": document.metadata.get("chunk_id"),
                    "snippet": snippet_text(document.page_content),
                }
                for document, _score in results
            ],
            citation_labels,
        )
        answer = "stub answer"
        if self.conversation_manager is not None:
            self.conversation_manager.create_session(session_id)
            self.conversation_manager.add_message(session_id, "user", question)
            self.conversation_manager.add_message(
                session_id,
                "assistant",
                answer,
                metadata={"citations": citations},
            )

        return {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "citations": citations,
        }


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        vector_db_dir=tmp_path / "vector_db",
        max_upload_mb=1,
        chunk_size=300,
        chunk_overlap=50,
        prune_orphan_vectors=False,
    )


@pytest.fixture
def pipeline(settings) -> FakePipeline:
    return FakePipeline(settings)


@pytest.fixture
def service(settings, pipeline) -> DocumentService:
    return DocumentService(
        registry=DocumentRegistry(db_path=settings.documents_db),
        vector_store=pipeline.vector_store,
        settings=settings,
        set_bm25_documents=pipeline.set_bm25_documents,
    )


# ---------------------------------------------------------------
# Safety net: the test run must never modify real project data.
# ---------------------------------------------------------------

def _real_data_fingerprint() -> dict:
    import hashlib
    from backend.core.config import PROJECT_ROOT

    files = [
        PROJECT_ROOT / "data" / "documents.db",
        PROJECT_ROOT / "data" / "conversations.db",
        PROJECT_ROOT / "vector_db" / "chroma.sqlite3",
    ]
    result = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.exists()
    }
    uploads = PROJECT_ROOT / "data" / "uploads"
    result["uploads"] = sorted(p.name for p in uploads.glob("*")) if uploads.exists() else []

    return result


@pytest.fixture(scope="session", autouse=True)
def real_data_must_not_change():
    before = _real_data_fingerprint()
    yield
    assert _real_data_fingerprint() == before, (
        "A test modified the real data/ or vector_db/ directory."
    )
