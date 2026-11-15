# Screenshots

Add real captures from your running app here. Do not commit mockups.

## How to capture

1. Start the backend and UI from the project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
streamlit run frontend/app.py
```

2. Open <http://127.0.0.1:8501> (or the port Streamlit prints).
3. Capture each page at roughly desktop width (≥1280px):

| File name | Page | What to show |
|---|---|---|
| `01-dashboard.png` | Dashboard | Live metrics, backend/Ollama status, recent docs/chats |
| `02-chat.png` | AI Chat | Document selector + one answered question with citations open |
| `03-library.png` | Document Library | Uploaded PDFs with status, pages and chunks |
| `04-history.png` | Chat History | At least one saved conversation |
| `05-evaluation.png` | RAG Evaluation | Latest run comparison table (after an evaluation run) |

4. Prefer PNG. Avoid uploading private PDFs in the frame if the repo will be public.
5. After saving files into this folder, reference them from the root `README.md` Screenshots section.
