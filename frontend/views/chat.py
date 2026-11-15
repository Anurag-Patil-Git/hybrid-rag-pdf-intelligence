"""Chat with one document, several documents, or every indexed document."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIError, get_client
from frontend.components import page_header, render_messages
from frontend.state import SUGGESTED_QUESTIONS, init_state

init_state()


def _sync_messages() -> None:
    session_id = st.session_state.active_session_id
    if st.session_state.loaded_session_id == session_id:
        return
    if not session_id:
        st.session_state.messages = []
        st.session_state.conversation_title = "New chat"
        return

    conversation = get_client().get_session(session_id)
    st.session_state.messages = conversation.get("messages") or []
    st.session_state.conversation_title = conversation.get("title") or "New chat"
    st.session_state.loaded_session_id = session_id


def _indexed_documents() -> list[dict]:
    documents = get_client().list_documents().get("documents", [])
    return [document for document in documents if document.get("status") == "indexed"]


def _start_new_chat() -> None:
    created = get_client().create_session("New chat")
    st.session_state.active_session_id = created["session_id"]
    st.session_state.conversation_title = created.get("title") or "New chat"
    st.session_state.messages = []
    st.session_state.loaded_session_id = created["session_id"]
    st.session_state.chat_error = None


def _send(question: str, indexed: list[dict]) -> None:
    search_all = st.session_state.search_scope != "selected"
    selected = [
        document_id
        for document_id in st.session_state.selected_document_ids
        if document_id in {item["document_id"] for item in indexed}
    ]

    if not search_all and not selected:
        st.session_state.chat_error = (
            "Select at least one document, or search all indexed documents."
        )
        return

    if not indexed and search_all:
        st.session_state.chat_error = "Upload a PDF before asking questions."
        return

    client = get_client()
    if not st.session_state.active_session_id:
        _start_new_chat()

    with st.spinner("Searching documents and writing an answer..."):
        result = client.chat(
            question,
            st.session_state.active_session_id,
            search_all=search_all,
            document_ids=selected,
        )

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer") or "",
        "metadata": {
            "citations": result.get("citations") or [],
            "rewritten_question": result.get("rewritten_question", ""),
            "retrieved_chunks": result.get("retrieved_chunks", 0),
            "selected_chunks": result.get("selected_chunks", 0),
            "context_characters": result.get("context_characters", 0),
        },
    })
    st.session_state.loaded_session_id = st.session_state.active_session_id
    st.session_state.chat_error = None


page_header(
    "Conversation",
    "AI Chat",
    "Answers stay grounded in the documents you select. Citations use the PDF name and page.",
)

try:
    indexed = _indexed_documents()
except APIError as error:
    st.error(str(error))
    indexed = []

try:
    _sync_messages()
except APIError as error:
    st.error(str(error))

title_col, new_col, rename_col, clear_col = st.columns([3.2, 1, 1, 1])
with title_col:
    st.subheader(st.session_state.conversation_title or "New chat")
with new_col:
    if st.button("New chat", use_container_width=True):
        try:
            _start_new_chat()
            st.rerun()
        except APIError as error:
            st.session_state.chat_error = str(error)
with rename_col:
    with st.popover("Rename", use_container_width=True):
        renamed = st.text_input("Title", value=st.session_state.conversation_title or "New chat")
        if st.button("Save title", disabled=not st.session_state.active_session_id):
            try:
                get_client().rename_session(st.session_state.active_session_id, renamed.strip())
                st.session_state.conversation_title = renamed.strip()
                st.rerun()
            except APIError as error:
                st.error(str(error))
with clear_col:
    if st.button("Clear", use_container_width=True, disabled=not st.session_state.active_session_id):
        st.session_state.confirm_clear_chat = True

if st.session_state.get("confirm_clear_chat"):
    st.warning("Clear every message in this conversation? The title is kept.")
    yes, no = st.columns(2)
    if yes.button("Clear messages", type="primary"):
        try:
            get_client().clear_session(st.session_state.active_session_id)
            st.session_state.messages = []
            st.session_state.loaded_session_id = st.session_state.active_session_id
            st.session_state.confirm_clear_chat = False
            st.rerun()
        except APIError as error:
            st.error(str(error))
    if no.button("Keep messages"):
        st.session_state.confirm_clear_chat = False
        st.rerun()

st.radio(
    "Documents to search",
    options=["all", "selected"],
    format_func=lambda value: (
        "All indexed documents" if value == "all" else "Selected documents"
    ),
    horizontal=True,
    key="search_scope",
)

names = {item["document_id"]: item["original_filename"] for item in indexed}
valid_ids = set(names)
current_ids = list(st.session_state.selected_document_ids)
filtered_ids = [document_id for document_id in current_ids if document_id in valid_ids]
if filtered_ids != current_ids:
    st.session_state.selected_document_ids = filtered_ids

if st.session_state.search_scope == "selected":
    if not indexed:
        st.info("No indexed documents yet. Upload a PDF in the Document Library.")
    else:
        st.multiselect(
            "Indexed documents",
            options=list(names),
            format_func=lambda document_id: names[document_id],
            key="selected_document_ids",
            placeholder="Choose one or more documents",
        )
elif not indexed:
    st.info("No indexed documents yet. Upload a PDF before asking a question.")
else:
    st.caption(f"Searching all {len(indexed)} indexed documents. New uploads are included automatically.")

if st.session_state.chat_error:
    st.error(st.session_state.chat_error)

render_messages(st.session_state.messages)

if not st.session_state.messages:
    st.caption("Suggested questions")
    suggestion_columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if suggestion_columns[index % 2].button(suggestion, key=f"suggest_{index}"):
            st.session_state.pending_question = suggestion
            st.rerun()

question = st.chat_input("Ask about the selected documents")
pending = st.session_state.pop("pending_question", None)
if pending:
    question = pending

if question:
    try:
        _send(question, indexed)
    except APIError as error:
        st.session_state.chat_error = str(error)
    st.rerun()
