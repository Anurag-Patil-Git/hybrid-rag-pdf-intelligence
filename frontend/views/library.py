"""Upload, inspect, replace, and delete PDFs."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIError, get_client
from frontend.components import (
    confirmed_button,
    empty_state,
    format_bytes,
    format_time,
    page_header,
    status_pill,
)

page_header(
    "Library",
    "Document Library",
    "Drop one or more PDFs here. Indexing runs on the server and the registry is updated when it finishes.",
)

try:
    listing = get_client().list_documents()
    documents = listing.get("documents", [])
except APIError as error:
    st.error(str(error))
    st.stop()

if st.session_state.get("notice"):
    st.success(st.session_state.notice)

uploaded = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Drag files onto this area or browse. Several PDFs can be indexed together.",
    key=f"library_uploader_{st.session_state.get('upload_cycle', 0)}",
)

if uploaded:
    for item in uploaded:
        st.caption(f"{item.name} · {format_bytes(item.size)}")
    if st.button("Index PDFs", type="primary"):
        successes = []
        failures = []
        client = get_client()
        for item in uploaded:
            try:
                result = client.upload_pdf(item.name, item.getvalue())
                document = result["document"]
                successes.append(
                    f"{document['original_filename']} "
                    f"({document['page_count']} pages, {document['chunk_count']} chunks)"
                )
            except APIError as error:
                failures.append(f"{item.name}: {error}")
        lines = []
        if successes:
            lines.append("Indexed " + "; ".join(successes) + ".")
        if failures:
            lines.append(" ".join(failures))
        st.session_state.notice = " ".join(lines) if lines else None
        st.session_state.upload_cycle = st.session_state.get("upload_cycle", 0) + 1
        st.rerun()

st.write("")
filter_col, status_col = st.columns([2, 1])
query = filter_col.text_input("Search documents", placeholder="Filter by file name")
status = status_col.selectbox(
    "Status",
    ["All", "indexed", "processing", "failed", "deleting"],
)

visible = documents
if query.strip():
    needle = query.strip().lower()
    visible = [
        document for document in visible
        if needle in document.get("original_filename", "").lower()
    ]
if status != "All":
    visible = [document for document in visible if document.get("status") == status]

st.caption(f"{len(visible)} of {len(documents)} documents")

if not documents:
    empty_state("Library is empty", "Upload a text-based PDF. Scanned pages without text are rejected.")
elif not visible:
    empty_state("No matches", "Try a different name or status filter.")
else:
    for document in visible:
        document_id = document["document_id"]
        with st.container(border=True):
            info, actions = st.columns([3, 1.4])
            with info:
                st.markdown(f"**{document['original_filename']}**")
                st.markdown(status_pill(document.get("status") or "unknown"), unsafe_allow_html=True)
                st.caption(
                    f"{format_bytes(document.get('file_size'))} · "
                    f"{document.get('page_count', 0)} pages · "
                    f"{document.get('chunk_count', 0)} chunks · "
                    f"Uploaded {format_time(document.get('created_at'))}"
                )
                if document.get("error_message"):
                    st.error(document["error_message"])
            with actions:
                if st.button("Details", key=f"details_{document_id}", use_container_width=True):
                    st.session_state.detail_document_id = (
                        None
                        if st.session_state.get("detail_document_id") == document_id
                        else document_id
                    )
                if st.button("Replace", key=f"replace_{document_id}", use_container_width=True):
                    st.session_state.replace_document_id = document_id

            if st.session_state.get("detail_document_id") == document_id:
                try:
                    record = get_client().get_document(document_id)
                except APIError as error:
                    st.error(str(error))
                else:
                    st.caption(
                        f"ID {record['document_id']} · stored as {record['stored_filename']}"
                    )
                    st.caption(
                        f"Updated {format_time(record.get('updated_at'))} · "
                        f"SHA-256 {record.get('file_hash', '')[:16]}…"
                    )

            if st.session_state.get("replace_document_id") == document_id:
                replacement = st.file_uploader(
                    "Replacement PDF",
                    type=["pdf"],
                    key=f"replace_file_{document_id}",
                )
                if replacement is not None:
                    st.caption(f"{replacement.name} · {format_bytes(replacement.size)}")
                    if confirmed_button(
                        "Replace this document",
                        "Confirm replace",
                        key=f"confirm_replace_{document_id}",
                    ):
                        try:
                            result = get_client().replace_pdf(
                                document_id,
                                replacement.name,
                                replacement.getvalue(),
                            )
                            changed = result.get("changed", True)
                            st.session_state.notice = (
                                result.get("message")
                                or (
                                    "Document replaced."
                                    if changed
                                    else "The file is identical, so nothing changed."
                                )
                            )
                            st.session_state.replace_document_id = None
                            st.rerun()
                        except APIError as error:
                            st.error(str(error))

            if confirmed_button("Delete", "Confirm delete", key=f"delete_{document_id}"):
                try:
                    get_client().delete_document(document_id)
                    if st.session_state.get("detail_document_id") == document_id:
                        st.session_state.detail_document_id = None
                    selected = st.session_state.get("selected_document_ids") or []
                    st.session_state.selected_document_ids = [
                        item for item in selected if item != document_id
                    ]
                    st.session_state.notice = f"Deleted {document['original_filename']}."
                    st.rerun()
                except APIError as error:
                    st.error(str(error))
