"""Saved conversations: search, resume, rename, and delete."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import APIError, get_client
from frontend.components import confirmed_button, empty_state, format_time, page_header
from frontend.navigation import open_chat
from frontend.state import init_state

init_state()

page_header(
    "History",
    "Chat History",
    "Resume a saved conversation. Renaming and deleting happen here and on the chat page.",
)

try:
    sessions = get_client().list_sessions().get("sessions", [])
except APIError as error:
    st.error(str(error))
    st.stop()

query = st.text_input("Search conversations", placeholder="Filter by title")
if query.strip():
    needle = query.strip().lower()
    sessions = [
        session for session in sessions
        if needle in (session.get("title") or "").lower()
    ]

if not sessions:
    empty_state(
        "No conversations to show",
        "Start a chat and it will appear here. Search matches the conversation title.",
    )
    st.stop()

for session in sessions:
    session_id = session["session_id"]
    active = session_id == st.session_state.active_session_id
    with st.container(border=True):
        info, actions = st.columns([3, 1.6])
        with info:
            prefix = "Current · " if active else ""
            st.markdown(f"**{prefix}{session.get('title') or 'New chat'}**")
            st.caption(
                f"{session.get('message_count', 0)} messages · "
                f"Updated {format_time(session.get('updated_at'))}"
            )
        with actions:
            if st.button("Resume", key=f"resume_{session_id}", use_container_width=True):
                open_chat(session_id)
            with st.popover("Rename"):
                title = st.text_input(
                    "Title",
                    value=session.get("title") or "New chat",
                    key=f"title_{session_id}",
                )
                if st.button("Save", key=f"save_{session_id}"):
                    try:
                        get_client().rename_session(session_id, title.strip())
                        if active:
                            st.session_state.conversation_title = title.strip()
                        st.rerun()
                    except APIError as error:
                        st.error(str(error))
            if confirmed_button("Delete", "Confirm delete", key=f"drop_{session_id}"):
                try:
                    get_client().delete_session(session_id)
                    if active:
                        st.session_state.active_session_id = None
                        st.session_state.loaded_session_id = None
                        st.session_state.messages = []
                        st.session_state.conversation_title = "New chat"
                    st.rerun()
                except APIError as error:
                    st.error(str(error))
