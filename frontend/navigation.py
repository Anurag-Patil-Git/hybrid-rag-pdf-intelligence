"""Page paths relative to ``frontend/app.py``."""

from __future__ import annotations

import streamlit as st

DASHBOARD_PAGE = "views/dashboard.py"
CHAT_PAGE = "views/chat.py"
LIBRARY_PAGE = "views/library.py"
HISTORY_PAGE = "views/history.py"
EVALUATION_PAGE = "views/evaluation.py"


def open_chat(session_id: str) -> None:
    """Resume a conversation and leave the current page."""

    st.session_state.active_session_id = session_id
    st.session_state.loaded_session_id = None
    st.session_state.chat_error = None
    st.switch_page(CHAT_PAGE)
