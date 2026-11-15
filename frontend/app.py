"""Multipage entrypoint. Run from the project root: streamlit run frontend/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components import render_sidebar_status
from frontend.navigation import (
    CHAT_PAGE,
    DASHBOARD_PAGE,
    EVALUATION_PAGE,
    HISTORY_PAGE,
    LIBRARY_PAGE,
)
from frontend.state import init_state
from frontend.styles import inject_css

st.set_page_config(
    page_title="Document Intelligence",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
init_state()

pages = [
    st.Page(DASHBOARD_PAGE, title="Dashboard", icon=":material/space_dashboard:", default=True),
    st.Page(CHAT_PAGE, title="AI Chat", icon=":material/forum:"),
    st.Page(LIBRARY_PAGE, title="Document Library", icon=":material/library_books:"),
    st.Page(HISTORY_PAGE, title="Chat History", icon=":material/history:"),
    st.Page(EVALUATION_PAGE, title="RAG Evaluation", icon=":material/analytics:"),
]

navigation = st.navigation(pages)
render_sidebar_status()
navigation.run()
