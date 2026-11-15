"""Session keys shared by every page.

Page changes must not wipe the active conversation or the document selection.
``init_state`` only fills keys that are not already set.
"""

from __future__ import annotations

import streamlit as st

# Must stay identical to backend.llm.NO_ANSWER.
NO_ANSWER = (
    "I couldn't find sufficient information in the uploaded "
    "document to answer this question."
)

SUGGESTED_QUESTIONS = [
    "What are the main topics in these documents?",
    "Summarize the key points and cite the page numbers.",
    "Which sections explain the most important terms?",
    "What should I read first, and why?",
]


def init_state() -> None:
    defaults = {
        "active_session_id": None,
        "loaded_session_id": None,
        "conversation_title": "New chat",
        "messages": [],
        "search_scope": "all",
        "selected_document_ids": [],
        "chat_error": None,
        "notice": None,
        "detail_document_id": None,
        "replace_document_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
