"""Dashboard. Every number comes from the API."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIError, get_client
from frontend.components import empty_state, format_time, page_header, show_pills, status_pill
from frontend.navigation import open_chat


def _load():
    client = get_client()
    health = client.health()
    documents = client.list_documents().get("documents", [])
    sessions = client.list_sessions().get("sessions", [])
    return health, documents, sessions


page_header(
    "Overview",
    "Dashboard",
    "Live counts from the document registry, the vector index, and saved conversations.",
)

try:
    health, documents, sessions = _load()
except APIError as error:
    st.error(str(error))
    st.stop()

first, second, third, fourth = st.columns(4)
first.metric("Documents", health.get("total_documents", 0))
second.metric("Indexed pages", health.get("total_pages", 0))
third.metric("Chunks", health.get("total_chunks", 0))
fourth.metric("Conversations", health.get("total_conversations", 0))

st.caption(f"{health.get('uploaded_documents', 0)} documents are indexed and searchable.")

show_pills([
    ("Backend connected" if health.get("backend_available") else "Backend unavailable",
     "ok" if health.get("backend_available") else "bad"),
    ("Ollama connected" if health.get("llm_available") else "Ollama unavailable",
     "ok" if health.get("llm_available") else "warn"),
    ("Vector database connected" if health.get("vector_db_available") else "Vector database unavailable",
     "ok" if health.get("vector_db_available") else "bad"),
])

st.write("")
docs_col, chats_col = st.columns(2)

with docs_col:
    st.subheader("Recent documents")
    if not documents:
        empty_state("No documents yet", "Upload a PDF from the Document Library to index it.")
    else:
        for document in documents[:5]:
            with st.container(border=True):
                st.markdown(f"**{document['original_filename']}**")
                st.markdown(status_pill(document.get("status") or "unknown"), unsafe_allow_html=True)
                st.caption(
                    f"{document.get('page_count', 0)} pages · "
                    f"{document.get('chunk_count', 0)} chunks · "
                    f"{format_time(document.get('created_at'))}"
                )

with chats_col:
    st.subheader("Recent conversations")
    if not sessions:
        empty_state("No conversations yet", "Start one from AI Chat. It will be saved automatically.")
    else:
        for session in sessions[:5]:
            with st.container(border=True):
                st.markdown(f"**{session.get('title') or 'New chat'}**")
                st.caption(
                    f"{session.get('message_count', 0)} messages · "
                    f"{format_time(session.get('updated_at'))}"
                )
                if st.button("Resume", key=f"dash_open_{session['session_id']}"):
                    open_chat(session["session_id"])
