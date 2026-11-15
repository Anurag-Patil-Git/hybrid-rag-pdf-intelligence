"""Visual theme shared by every page."""

from __future__ import annotations

import streamlit as st

CSS = """
:root {
  --bg: #10141C;
  --panel: #1B2230;
  --panel-2: #212A3C;
  --line: #2E3A52;
  --text: #E7ECF5;
  --muted: #9AA6BD;
  --blue: #7EAEFF;
  --violet: #A99BFF;
  --ok: #3DDC97;
  --warn: #E6B35A;
  --bad: #F07178;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", Inter, ui-sans-serif, system-ui, sans-serif;
}

[data-testid="stHeader"] {
  background: rgba(16, 20, 28, 0.92);
  border-bottom: 1px solid var(--line);
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 1120px;
}

section[data-testid="stSidebar"] {
  background: #141A26;
  border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
  border-radius: 8px;
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--violet);
  border-radius: 12px;
  padding: 0.85rem 0.95rem 0.7rem;
}

div[data-testid="stMetric"] label {
  color: var(--muted);
}

[data-testid="stChatMessage"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.35rem 0.7rem 0.15rem;
  margin-bottom: 0.35rem;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: #1A2438;
  border-color: #3A4D78;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel);
  border-color: var(--line) !important;
  border-radius: 12px;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 8px;
  border: 1px solid var(--line);
}

div[data-testid="stExpander"] {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 10px;
}

.pill {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  letter-spacing: 0.01em;
  border: 1px solid var(--line);
  color: var(--muted);
  margin-right: 0.35rem;
}

.pill.ok { color: var(--ok); border-color: #245C46; background: #13261E; }
.pill.warn { color: var(--warn); border-color: #6A5424; background: #2A2416; }
.pill.bad { color: var(--bad); border-color: #6A3036; background: #2A171B; }
.pill.info { color: var(--blue); border-color: #2C4A78; background: #152033; }

.page-kicker {
  color: var(--violet);
  font-size: 0.78rem;
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.15rem;
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
