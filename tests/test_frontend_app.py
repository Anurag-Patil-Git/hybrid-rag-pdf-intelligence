"""Streamlit AppTest checks. Mocked API; no live backend or model downloads."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import APIError

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


class StubClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def _raise(self):
        raise APIError("The API is unreachable. Start the backend and try again.")

    def health(self):
        if self.fail:
            self._raise()
        return {
            "status": "degraded",
            "backend_available": True,
            "vector_db_available": True,
            "llm_available": False,
            "pipeline_initialized": True,
            "uploaded_documents": 1,
            "total_documents": 1,
            "total_pages": 2,
            "total_chunks": 6,
            "total_conversations": 1,
            "orphan_vector_sources": 0,
        }

    def list_documents(self):
        if self.fail:
            self._raise()
        return {
            "total_documents": 1,
            "documents": [{
                "document_id": "doc-1",
                "original_filename": "notes.pdf",
                "stored_filename": "x_notes.pdf",
                "file_size": 1234,
                "page_count": 2,
                "chunk_count": 6,
                "status": "indexed",
                "error_message": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "file_hash": "abc",
            }],
        }

    def list_sessions(self):
        if self.fail:
            self._raise()
        return {
            "total_sessions": 1,
            "sessions": [{
                "session_id": "sess-1",
                "title": "Saved chat",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "message_count": 2,
            }],
        }

    def get_session(self, session_id):
        if self.fail:
            self._raise()
        return {
            "session_id": session_id,
            "title": "Saved chat",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "messages": [
                {"role": "user", "content": "Hello", "metadata": {}},
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "metadata": {
                        "citations": [{
                            "original_filename": "notes.pdf",
                            "page": 1,
                            "snippet": "snippet text",
                            "source": "x_notes.pdf",
                        }],
                        "retrieved_chunks": 3,
                        "selected_chunks": 1,
                        "context_characters": 40,
                        "rewritten_question": "Hello",
                    },
                },
            ],
        }

    def create_session(self, title="New chat"):
        return {"session_id": "new-sess", "title": title}

    def rename_session(self, session_id, title):
        return {"session_id": session_id, "title": title}

    def clear_session(self, session_id):
        return {"session_id": session_id}

    def delete_session(self, session_id):
        return {"session_id": session_id}

    def get_document(self, document_id):
        return self.list_documents()["documents"][0]

    def upload_pdf(self, *a, **k):
        return {"document": self.list_documents()["documents"][0]}

    def replace_pdf(self, *a, **k):
        return {"changed": True, "message": "ok"}

    def delete_document(self, *a, **k):
        return {"message": "ok"}

    def chat(self, *a, **k):
        raise AssertionError("chat not needed for page load")


def _patch_client(monkeypatch, client):
    import frontend.api_client as api_mod
    import frontend.components as components

    monkeypatch.setattr(api_mod, "get_client", lambda: client)
    monkeypatch.setattr(components, "get_client", lambda: client)


def test_all_five_pages_load_without_exceptions(monkeypatch):
    _patch_client(monkeypatch, StubClient())
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Dashboard"]

    for path, expected in [
        ("views/dashboard.py", "Dashboard"),
        ("views/chat.py", "AI Chat"),
        ("views/library.py", "Document Library"),
        ("views/history.py", "Chat History"),
        ("views/evaluation.py", "RAG Evaluation"),
    ]:
        at.switch_page(path)
        at.run()
        assert not at.exception, path
        assert expected in [t.value for t in at.title]


def test_navigation_preserves_active_conversation_and_selection(monkeypatch):
    _patch_client(monkeypatch, StubClient())
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.switch_page("views/chat.py")
    at.run()
    at.session_state.active_session_id = "sess-1"
    at.session_state.loaded_session_id = None
    at.session_state.selected_document_ids = ["doc-1"]
    at.session_state.search_scope = "selected"
    at.run()
    assert at.session_state.conversation_title == "Saved chat"
    assert len(at.session_state.messages) == 2
    assert len(at.chat_message) == 2

    at.switch_page("views/library.py")
    at.run()
    at.switch_page("views/chat.py")
    at.run()
    assert at.session_state.active_session_id == "sess-1"
    assert at.session_state.selected_document_ids == ["doc-1"]
    assert len(at.session_state.messages) == 2


def test_unreachable_api_shows_errors_without_crashing(monkeypatch):
    _patch_client(monkeypatch, StubClient(fail=True))
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.error) >= 1

    for path in [
        "views/dashboard.py",
        "views/chat.py",
        "views/library.py",
        "views/history.py",
        "views/evaluation.py",
    ]:
        at.switch_page(path)
        at.run()
        assert not at.exception, path


def test_evaluation_page_reads_saved_results(monkeypatch):
    _patch_client(monkeypatch, StubClient())
    runs = sorted(Path("evaluation/results").glob("*.json"))
    if not runs:
        pytest.skip("No saved evaluation runs to render.")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.switch_page("views/evaluation.py")
    at.run()
    assert not at.exception
    assert "RAG Evaluation" in [t.value for t in at.title]
    # The page should expose run metrics rather than the empty-state copy.
    body = " ".join(str(getattr(item, "value", item)) for item in at.markdown)
    assert "No evaluation results yet" not in body
    assert any(label in [m.label for m in at.metric] for label in (
        "Verified questions",
        "Top-K",
        "Kind",
    ))
