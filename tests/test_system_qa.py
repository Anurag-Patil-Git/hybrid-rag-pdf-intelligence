"""Phase 6 hermetic end-to-end workflows. Temporary dirs only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.conversation import ConversationManager
from backend.evaluation.dataset import load_dataset, verified_questions
from backend.evaluation.results import list_runs, load_run, save_run
from backend.evaluation.runner import run_retrieval_experiment
from backend.evaluation.retrieval_modes import RetrievalBundle
from backend.bm25_retriever import BM25Retriever
from backend.main import create_app
from frontend.api_client import APIError, build_chat_payload
from tests.conftest import FakePipeline, GEMINI_PDF, LANGCHAIN_PDF, make_pdf


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr("backend.main.check_ollama", lambda *_a, **_k: False)
    app = create_app(
        settings=settings,
        pipeline_factory=lambda s, conversations: FakePipeline(s, conversations),
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, name, data):
    return client.post("/upload", files={"file": (name, data, "application/pdf")})


# ---------------- Workflow A: document lifecycle ----------------

def test_workflow_a_document_lifecycle(client):
    uploaded = _upload(client, "life.pdf", LANGCHAIN_PDF)
    assert uploaded.status_code == 201
    document = uploaded.json()["document"]
    assert document["status"] == "indexed"
    assert document["page_count"] > 0
    assert document["chunk_count"] > 0

    listing = client.get("/documents").json()
    assert listing["total_documents"] == 1
    assert listing["documents"][0]["document_id"] == document["document_id"]

    duplicate = _upload(client, "copy.pdf", LANGCHAIN_PDF)
    assert duplicate.status_code == 409

    replacement = make_pdf("Replacement content about transformers and attention. " * 10)
    replaced = client.put(
        f"/documents/{document['document_id']}",
        files={"file": ("life2.pdf", replacement, "application/pdf")},
    )
    assert replaced.status_code == 200
    assert replaced.json()["changed"] is True
    got = client.get(f"/documents/{document['document_id']}").json()
    assert got["original_filename"] == "life2.pdf"
    assert got["status"] == "indexed"

    deleted = client.delete(f"/documents/{document['document_id']}")
    assert deleted.status_code == 200
    assert client.get(f"/documents/{document['document_id']}").status_code == 404
    assert client.get("/documents").json()["total_documents"] == 0

    gone = client.post(
        "/chat",
        json={
            "question": "anything",
            "session_id": "s-life",
            "document_ids": [document["document_id"]],
        },
    )
    assert gone.status_code == 404


# ---------------- Workflow B: multi-document chat ----------------

def test_workflow_b_multi_document_chat(client):
    first = _upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]
    second = _upload(client, "b.pdf", GEMINI_PDF).json()["document"]

    only_a = client.post(
        "/chat",
        json={
            "question": "retriever documents",
            "session_id": "s-a",
            "document_ids": [first["document_id"]],
        },
    ).json()
    assert {c["document_id"] for c in only_a["citations"]} == {first["document_id"]}

    both = client.post(
        "/chat",
        json={
            "question": "retriever documents",
            "session_id": "s-both",
            "document_ids": [first["document_id"], second["document_id"]],
        },
    ).json()
    assert {c["document_id"] for c in both["citations"]} <= {
        first["document_id"],
        second["document_id"],
    }

    all_docs = client.post(
        "/chat",
        json={"question": "retriever documents", "session_id": "s-all"},
    )
    assert all_docs.status_code == 200

    empty = client.post(
        "/chat",
        json={"question": "x", "session_id": "s-empty", "document_ids": []},
    )
    assert empty.status_code == 422

    assert "document_ids" not in build_chat_payload("q", "s", search_all=True)
    with pytest.raises(APIError):
        build_chat_payload("q", "s", search_all=False, document_ids=[])


# ---------------- Workflow C: conversation ----------------

def test_workflow_c_conversation_persistence(client, settings):
    created = client.post("/sessions", json={"title": "QA Chat"}).json()
    session_id = created["session_id"]

    _upload(client, "notes.pdf", LANGCHAIN_PDF)
    asked = client.post(
        "/chat",
        json={"question": "retriever documents", "session_id": session_id},
    )
    assert asked.status_code == 200

    history = client.get(f"/sessions/{session_id}").json()
    assert history["title"] == "QA Chat"
    assert len(history["messages"]) >= 2

    renamed = client.patch(f"/sessions/{session_id}", json={"title": "Renamed QA"})
    assert renamed.status_code == 200
    assert client.get(f"/sessions/{session_id}").json()["title"] == "Renamed QA"

    cleared = client.delete(f"/sessions/{session_id}/messages")
    assert cleared.status_code == 200
    after = client.get(f"/sessions/{session_id}").json()
    assert after["title"] == "Renamed QA"
    assert after["messages"] == []

    # Direct SQLite persistence across manager instances
    manager = ConversationManager(db_path=settings.conversations_db)
    assert manager.get_session(session_id)["title"] == "Renamed QA"

    assert client.delete(f"/sessions/{session_id}").status_code == 200
    assert client.get(f"/sessions/{session_id}").status_code == 404


# ---------------- Workflow D: evaluation ----------------

def test_workflow_d_evaluation_results_are_preserved(tmp_path, settings, pipeline):
    dataset = {
        "dataset_version": "qa-synth",
        "documents": [{
            "doc_key": "notes",
            "match_original_filename": "notes.pdf",
        }],
        "questions": [{
            "question_id": "q1",
            "question": "indexes retrieve documents",
            "verified": True,
            "expected_answer_criteria": "indexes",
            "relevant": [{"doc_key": "notes", "page": 1, "chunk_ids": None}],
        }],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    # Tiny corpus already in FakePipeline after service upload via DocumentService
    from backend.document_registry import DocumentRegistry
    from backend.services.document_service import DocumentService

    service = DocumentService(
        registry=DocumentRegistry(db_path=settings.documents_db),
        vector_store=pipeline.vector_store,
        settings=settings,
        set_bm25_documents=pipeline.set_bm25_documents,
    )
    record = service.upload("notes.pdf", LANGCHAIN_PDF)
    chunks = [
        chunk
        for values in service._chunks_by_source.values()
        for chunk in values
    ]
    bundle = RetrievalBundle(
        vector_store=pipeline.vector_store,
        bm25_retriever=BM25Retriever(documents=chunks) if chunks else None,
        reranker=None,
    )
    results_dir = tmp_path / "results"
    first = run_retrieval_experiment(
        dataset_path=dataset_path,
        bundle=bundle,
        document_records=[record],
        top_k=3,
        modes=("dense", "bm25", "hybrid"),
        results_directory=results_dir,
        save=True,
    )
    second = run_retrieval_experiment(
        dataset_path=dataset_path,
        bundle=bundle,
        document_records=[record],
        top_k=3,
        modes=("hybrid",),
        limit=1,
        results_directory=results_dir,
        save=True,
    )
    assert first["run_id"] != second["run_id"]
    runs = list_runs(results_dir)
    assert len(runs) == 2
    assert Path(first["saved_to"]).exists()
    assert Path(second["saved_to"]).exists()
    reloaded = load_run(first["saved_to"])
    assert reloaded["dataset_version"] == "qa-synth"
    assert set(reloaded["results"]) == {"dense", "bm25", "hybrid"}


def test_capstone_dataset_still_loads_for_dashboard():
    path = Path("evaluation/datasets/capstone_v1.json")
    dataset = load_dataset(path)
    assert len(verified_questions(dataset)) >= 30
