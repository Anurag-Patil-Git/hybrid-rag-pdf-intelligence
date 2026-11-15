"""HTTP client for the FastAPI backend.

Pages should call this module instead of ``requests`` directly.
"""

from __future__ import annotations

import os

import requests
import streamlit as st


class APIError(Exception):
    """A failed API call, with a message that is safe to show in the UI."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def build_chat_payload(
    question: str,
    session_id: str,
    *,
    search_all: bool,
    document_ids: list[str] | None = None,
) -> dict:
    """Build a ``POST /chat`` body.

    Searching every indexed document omits ``document_ids``. An explicit
    selection sends those IDs. An empty selection is rejected here so the
    client never posts ``document_ids: []``.
    """

    question = (question or "").strip()
    session_id = (session_id or "").strip()

    if not question:
        raise APIError("Enter a question.", 422)
    if not session_id:
        raise APIError("Start or resume a conversation first.", 422)

    payload = {"question": question, "session_id": session_id}
    if search_all:
        return payload

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in document_ids or []:
        if not isinstance(value, str) or not value.strip():
            raise APIError("Each document ID must be a non-empty string.", 422)
        document_id = value.strip()
        if document_id not in seen:
            seen.add(document_id)
            cleaned.append(document_id)

    if not cleaned:
        raise APIError(
            "Select at least one document, or search all indexed documents.",
            422,
        )

    payload["document_ids"] = cleaned
    return payload


def message_from_response(response: requests.Response) -> str:
    """Turn an error response into a short sentence. Never includes a traceback."""

    try:
        body = response.json()
    except ValueError:
        return f"Request failed ({response.status_code})."

    detail = body.get("detail") if isinstance(body, dict) else None

    if isinstance(detail, dict):
        message = str(detail.get("message") or "Request failed.")
        existing = detail.get("existing_document") or {}
        name = existing.get("original_filename") if isinstance(existing, dict) else None
        if name:
            message = f"{message} Existing document: {name}."
        return message

    if isinstance(detail, list):
        return "The request was rejected. Check the values and try again."

    if detail:
        return str(detail)

    return f"Request failed ({response.status_code})."


class APIClient:
    def __init__(self, base_url: str, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5)

    def list_documents(self) -> dict:
        return self._request("GET", "/documents")

    def get_document(self, document_id: str) -> dict:
        return self._request("GET", f"/documents/{document_id}")

    def upload_pdf(self, filename: str, data: bytes) -> dict:
        return self._request(
            "POST",
            "/upload",
            files={"file": (filename, data, "application/pdf")},
            timeout=300,
        )

    def replace_pdf(self, document_id: str, filename: str, data: bytes) -> dict:
        return self._request(
            "PUT",
            f"/documents/{document_id}",
            files={"file": (filename, data, "application/pdf")},
            timeout=300,
        )

    def delete_document(self, document_id: str) -> dict:
        return self._request("DELETE", f"/documents/{document_id}")

    def list_sessions(self) -> dict:
        return self._request("GET", "/sessions")

    def get_session(self, session_id: str) -> dict:
        return self._request("GET", f"/sessions/{session_id}")

    def create_session(self, title: str = "New chat") -> dict:
        return self._request("POST", "/sessions", json={"title": title})

    def rename_session(self, session_id: str, title: str) -> dict:
        return self._request("PATCH", f"/sessions/{session_id}", json={"title": title})

    def delete_session(self, session_id: str) -> dict:
        return self._request("DELETE", f"/sessions/{session_id}")

    def clear_session(self, session_id: str) -> dict:
        return self._request("DELETE", f"/sessions/{session_id}/messages")

    def chat(
        self,
        question: str,
        session_id: str,
        *,
        search_all: bool,
        document_ids: list[str] | None = None,
    ) -> dict:
        payload = build_chat_payload(
            question,
            session_id,
            search_all=search_all,
            document_ids=document_ids,
        )
        return self._request("POST", "/chat", json=payload, timeout=300)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout,
                **kwargs,
            )
        except requests.RequestException:
            raise APIError(
                "The API is unreachable. Start the backend and try again."
            ) from None

        if response.ok:
            return response.json()

        raise APIError(message_from_response(response), response.status_code)


def get_client() -> APIClient:
    if "api_client" not in st.session_state:
        base_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
        st.session_state.api_client = APIClient(base_url)
    return st.session_state.api_client
