"""Reusable layout pieces. Labels passed to HTML are fixed strings, not user text."""

from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import APIError, get_client
from frontend.state import NO_ANSWER


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<p class="page-kicker">{kicker}</p>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def empty_state(title: str, body: str) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.markdown(body)


def format_bytes(size: int | None) -> str:
    if size is None:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def format_time(value: str | None) -> str:
    if not value:
        return "—"
    return value.replace("T", " ")[:16]


def status_pill(status: str) -> str:
    kind = {
        "indexed": "ok",
        "processing": "info",
        "deleting": "warn",
        "failed": "bad",
        "healthy": "ok",
        "degraded": "warn",
        "unavailable": "bad",
    }.get(status, "info")
    label = html.escape(status or "unknown")
    return f'<span class="pill {kind}">{label}</span>'


def show_pills(items: list[tuple[str, str]]) -> None:
    """Render status pills. ``items`` is ``(label, kind)`` with kind ok/warn/bad/info."""

    safe_kinds = {"ok", "warn", "bad", "info"}
    html_parts = " ".join(
        f'<span class="pill {kind if kind in safe_kinds else "info"}">{html.escape(label)}</span>'
        for label, kind in items
    )
    st.markdown(html_parts, unsafe_allow_html=True)


def confirmed_button(label: str, confirm_label: str, key: str) -> bool:
    """Two-step button. Returns True only on the confirming click."""

    stage_key = f"{key}_stage"
    if st.session_state.get(stage_key) != "confirm":
        if st.button(label, key=f"{key}_ask"):
            st.session_state[stage_key] = "confirm"
            st.rerun()
        return False

    st.caption("Confirm this action. It cannot be undone.")
    yes, no = st.columns(2)
    if yes.button(confirm_label, key=f"{key}_yes", type="primary"):
        st.session_state[stage_key] = "idle"
        return True
    if no.button("Cancel", key=f"{key}_no"):
        st.session_state[stage_key] = "idle"
        st.rerun()
    return False


def render_citations(citations: list[dict] | None) -> None:
    if not citations:
        return

    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for citation in citations:
            name = citation.get("original_filename") or citation.get("source") or "Document"
            page = citation.get("page", "—")
            st.markdown(f"**{name}**  \nPage {page}")
            snippet = citation.get("snippet")
            if snippet:
                with st.expander("Text snippet", expanded=False):
                    st.text(snippet)


def render_retrieval(metadata: dict | None) -> None:
    if not metadata:
        return

    has_detail = any(
        metadata.get(key) not in (None, "", 0)
        for key in (
            "rewritten_question",
            "retrieved_chunks",
            "selected_chunks",
            "context_characters",
        )
    )
    if not has_detail and "retrieved_chunks" not in metadata:
        return

    with st.expander("Retrieval details", expanded=False):
        rewritten = metadata.get("rewritten_question")
        if rewritten:
            st.caption("Rewritten question")
            st.write(rewritten)
        first, second, third = st.columns(3)
        first.metric("Retrieved", metadata.get("retrieved_chunks", 0))
        second.metric("Selected", metadata.get("selected_chunks", 0))
        third.metric("Context characters", metadata.get("context_characters", 0))


def render_messages(messages: list[dict]) -> None:
    if not messages:
        empty_state(
            "Ask a question",
            "Choose the documents to search, then ask something about them. "
            "Answers cite the PDF name and page.",
        )
        return

    for message in messages:
        role = message.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            content = message.get("content") or ""
            if role == "assistant" and content.strip() == NO_ANSWER:
                st.warning(content)
            else:
                st.markdown(content)
            if role == "assistant":
                metadata = message.get("metadata") or {}
                render_citations(metadata.get("citations"))
                render_retrieval(metadata)


def render_sidebar_status() -> None:
    with st.sidebar:
        st.divider()
        try:
            health = get_client().health()
        except APIError as error:
            st.error(str(error))
            return

        backend = "ok" if health.get("backend_available") else "bad"
        model = "ok" if health.get("llm_available") else "warn"
        vectors = "ok" if health.get("vector_db_available") else "bad"
        show_pills([
            ("Backend", backend),
            ("Ollama", model),
            ("Vectors", vectors),
        ])
        st.caption(
            f"{health.get('total_documents', 0)} documents · "
            f"{health.get('total_chunks', 0)} chunks"
        )
