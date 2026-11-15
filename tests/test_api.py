"""Phase 1: HTTP behaviour of the document and chat endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.conftest import GEMINI_PDF, LANGCHAIN_PDF, FakePipeline


@pytest.fixture(autouse=True)
def ollama_down(monkeypatch):
    """Health checks must not depend on a running Ollama server."""

    monkeypatch.setattr("backend.main.check_ollama", lambda *_args, **_kwargs: False)


@pytest.fixture
def client(settings):
    app = create_app(
        settings=settings,
        pipeline_factory=lambda s, conversations: FakePipeline(s, conversations),
    )

    with TestClient(app) as test_client:  # runs startup (reconcile)
        yield test_client


def upload(client, name, data):
    return client.post("/upload", files={"file": (name, data, "application/pdf")})


def test_health_on_empty_system(client):
    body = client.get("/health").json()

    # The API and its temporary Chroma store are up. Ollama is not, so the
    # process is degraded rather than healthy or unavailable.
    assert body["status"] == "degraded"
    assert body["backend_available"] is True
    assert body["vector_db_available"] is True
    assert body["llm_available"] is False
    assert body["pipeline_initialized"] is True
    assert body["uploaded_documents"] == 0
    assert body["total_documents"] == 0
    assert body["total_pages"] == 0
    assert body["total_chunks"] == 0
    assert body["total_conversations"] == 0
    assert body["orphan_vector_sources"] == 0


def test_health_is_healthy_only_when_ollama_answers(client, monkeypatch):
    monkeypatch.setattr("backend.main.check_ollama", lambda *_args, **_kwargs: True)

    document = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]
    client.post("/sessions", json={"title": "Notes"})

    body = client.get("/health").json()

    assert body["status"] == "healthy"
    assert body["backend_available"] is True
    assert body["vector_db_available"] is True
    assert body["llm_available"] is True
    assert body["total_pages"] == document["page_count"]
    assert body["total_chunks"] == document["chunk_count"]
    assert body["total_conversations"] == 1


def test_vector_db_outage_degrades_without_hiding_the_backend(client, monkeypatch):
    monkeypatch.setattr("backend.main.check_ollama", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("backend.main.vector_store_reachable", lambda _store: False)

    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["backend_available"] is True
    assert body["vector_db_available"] is False
    assert body["llm_available"] is True


def test_registry_outage_marks_backend_unavailable(client, monkeypatch):
    def explode(self):
        raise RuntimeError("secret registry failure")

    monkeypatch.setattr(
        "backend.services.document_service.DocumentService.stats", explode
    )
    monkeypatch.setattr("backend.main.check_ollama", lambda *_args, **_kwargs: True)

    response = client.get("/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "unavailable"
    assert body["backend_available"] is False
    assert body["llm_available"] is True
    assert "secret" not in response.text


def test_upload_list_get(client):
    response = upload(client, "a.pdf", LANGCHAIN_PDF)

    assert response.status_code == 201
    document = response.json()["document"]
    assert document["status"] == "indexed"

    listing = client.get("/documents").json()
    assert listing["total_documents"] == 1
    assert listing["documents"][0]["document_id"] == document["document_id"]
    assert client.get(f"/documents/{document['document_id']}").json() == document
    assert client.get("/health").json()["total_chunks"] == document["chunk_count"]


def test_duplicate_upload_returns_409_with_existing_document(client):
    first = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]

    response = upload(client, "copy.pdf", LANGCHAIN_PDF)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["existing_document"]["document_id"] == first["document_id"]
    assert client.get("/documents").json()["total_documents"] == 1


@pytest.mark.parametrize(
    "name, data, status",
    [
        ("a.txt", b"%PDF-1.4", 400),
        ("a.pdf", b"hello", 400),
        ("a.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 10), 413),
        ("a.pdf", b"%PDF-1.4 corrupt", 422),
    ],
)
def test_rejected_uploads(client, name, data, status):
    assert upload(client, name, data).status_code == status
    assert client.get("/documents").json()["total_documents"] == 0


def test_delete_document(client):
    document = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]

    assert client.delete(f"/documents/{document['document_id']}").status_code == 200
    assert client.get(f"/documents/{document['document_id']}").status_code == 404
    assert client.delete(f"/documents/{document['document_id']}").status_code == 404
    assert client.get("/documents").json()["total_documents"] == 0


def test_replace_document(client):
    document = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]
    url = f"/documents/{document['document_id']}"

    response = client.put(url, files={"file": ("b.pdf", GEMINI_PDF, "application/pdf")})

    assert response.status_code == 200
    assert response.json()["changed"] is True
    assert client.get(url).json()["original_filename"] == "b.pdf"

    same = client.put(url, files={"file": ("b.pdf", GEMINI_PDF, "application/pdf")})
    assert same.json()["changed"] is False

    assert client.put("/documents/nope", files={"file": ("x.pdf", GEMINI_PDF, "application/pdf")}).status_code == 404


def test_chat_requires_an_indexed_document(client):
    response = client.post("/chat", json={"question": "hi", "session_id": "s1"})

    assert response.status_code == 409


def test_chat_searches_only_indexed_documents(client):
    a = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]
    b = upload(client, "b.pdf", GEMINI_PDF).json()["document"]

    everything = client.post(
        "/chat", json={"question": "retriever documents", "session_id": "s1"}
    )
    assert everything.status_code == 200

    only_b = client.post(
        "/chat",
        json={"question": "retriever documents", "session_id": "s1", "source": b["stored_filename"]},
    ).json()
    assert {c["source"] for c in only_b["citations"]} == {b["stored_filename"]}

    client.delete(f"/documents/{b['document_id']}")

    gone = client.post(
        "/chat",
        json={"question": "anything", "session_id": "s1", "source": b["stored_filename"]},
    )
    assert gone.status_code == 404

    remaining = client.post("/chat", json={"question": "anything", "session_id": "s1"}).json()
    assert {c["source"] for c in remaining["citations"]} <= {a["stored_filename"]}


def test_chat_selects_documents_by_id(client):
    first = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]
    second = upload(client, "b.pdf", GEMINI_PDF).json()["document"]

    only_first = client.post(
        "/chat",
        json={
            "question": "retriever documents",
            "session_id": "s1",
            "document_ids": [first["document_id"]],
        },
    )
    assert only_first.status_code == 200
    body = only_first.json()
    assert body["citations"]
    assert {item["source"] for item in body["citations"]} == {first["stored_filename"]}
    assert {item["document_id"] for item in body["citations"]} == {first["document_id"]}
    assert {item["original_filename"] for item in body["citations"]} == {"a.pdf"}
    assert all(item["snippet"] for item in body["citations"])

    both = client.post(
        "/chat",
        json={
            "question": "retriever documents",
            "session_id": "s2",
            "document_ids": [second["document_id"], first["document_id"]],
        },
    ).json()
    assert first["document_id"] in {item["document_id"] for item in both["citations"]}
    assert {item["document_id"] for item in both["citations"]} <= {
        first["document_id"],
        second["document_id"],
    }
    assert second["stored_filename"] not in {
        item["source"] for item in only_first.json()["citations"]
    }


def test_chat_rejects_an_empty_or_conflicting_selection(client):
    first = upload(client, "a.pdf", LANGCHAIN_PDF).json()["document"]
    second = upload(client, "b.pdf", GEMINI_PDF).json()["document"]

    empty = client.post(
        "/chat",
        json={"question": "anything", "session_id": "s1", "document_ids": []},
    )
    assert empty.status_code == 422
    assert "document" in empty.json()["detail"].lower()

    missing = client.post(
        "/chat",
        json={"question": "anything", "session_id": "s1", "document_ids": ["missing-id"]},
    )
    assert missing.status_code == 404

    conflict = client.post(
        "/chat",
        json={
            "question": "anything",
            "session_id": "s1",
            "source": first["stored_filename"],
            "document_ids": [second["document_id"]],
        },
    )
    assert conflict.status_code == 422
    assert "different documents" in conflict.json()["detail"]

    agreed = client.post(
        "/chat",
        json={
            "question": "retriever documents",
            "session_id": "s1",
            "source": first["stored_filename"],
            "document_ids": [first["document_id"]],
        },
    )
    assert agreed.status_code == 200
    assert {item["document_id"] for item in agreed.json()["citations"]} == {first["document_id"]}

    client.delete(f"/documents/{second['document_id']}")
    gone = client.post(
        "/chat",
        json={
            "question": "anything",
            "session_id": "s1",
            "document_ids": [second["document_id"]],
        },
    )
    assert gone.status_code == 404


def test_chat_error_does_not_expose_internal_details(settings):
    class BoomPipeline(FakePipeline):
        def ask(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    app = create_app(
        settings=settings,
        pipeline_factory=lambda s, conversations: BoomPipeline(s, conversations),
    )

    with TestClient(app) as boom_client:
        document = upload(boom_client, "a.pdf", LANGCHAIN_PDF)
        assert document.status_code == 201

        response = boom_client.post(
            "/chat", json={"question": "What is a retriever?", "session_id": "s1"}
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Question processing failed."
    assert "secret" not in response.text


def test_clear_session_removes_messages_and_keeps_the_title(client):
    created = client.post("/sessions", json={"title": "Notes"}).json()
    session_id = created["session_id"]
    manager = client.app.state.conversation_manager
    manager.add_message(session_id, "user", "What is a retriever?")
    manager.add_message(
        session_id,
        "assistant",
        "It finds documents.",
        metadata={"citations": []},
    )

    response = client.delete(f"/sessions/{session_id}/messages")

    assert response.status_code == 200
    body = client.get(f"/sessions/{session_id}").json()
    assert body["title"] == "Notes"
    assert body["messages"] == []
    assert client.delete("/sessions/missing-session/messages").status_code == 404


def test_session_endpoints_still_work(client):
    created = client.post("/sessions", json={"title": "First"}).json()
    session_id = created["session_id"]

    assert client.patch(f"/sessions/{session_id}", json={"title": "Renamed"}).status_code == 200
    assert client.get(f"/sessions/{session_id}").json()["title"] == "Renamed"
    assert client.get("/sessions").json()["total_sessions"] == 1
    assert client.delete(f"/sessions/{session_id}").status_code == 200
    assert client.get(f"/sessions/{session_id}").status_code == 404
