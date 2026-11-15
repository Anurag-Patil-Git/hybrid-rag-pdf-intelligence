# Proposed Git commit plan

**Do not push until the owner explicitly approves.**

A local `git init` was performed for ignore auditing. No commits and no remotes have been created by this plan.

## Leave untracked (never commit)

| Path | Reason |
|---|---|
| `.env` | Secrets / local config (empty today, still ignored) |
| `data/documents.db`, `data/conversations.db` | Runtime SQLite |
| `data/uploads/*.pdf` | User / study PDFs |
| `vector_db/**` (except `.gitkeep`) | Chroma persistence |
| `evaluation/results/*.json` | Contains retrieved chunk text + local registry IDs |
| `__pycache__/`, `.pytest_cache/` | Generated |
| `docs/screenshots/*.png` | Capture locally; ignore until you choose to publish images |

## Optional omit from first public commit

| Path | Reason |
|---|---|
| `requirements-freeze-legacy.txt` | Full historical freeze; includes unrelated packages (e.g. `openai`) |
| `requirements-rag.txt` | Unpinned legacy list; superseded by `requirements.txt` |

Keep them locally or move to a private archive. Prefer installing from `requirements.txt` + `requirements-dev.txt` only.

## Suggested commit sequence (after approval)

### 1. `chore: add gitignore and env example`

- `.gitignore`
- `.env.example`
- `data/uploads/.gitkeep`
- `vector_db/.gitkeep`
- `evaluation/results/.gitkeep`
- `evaluation/manual_reviews/.gitkeep`

### 2. `feat: add RAG backend, Streamlit UI, and evaluation harness`

- `backend/`
- `frontend/`
- `.streamlit/config.toml`
- `evaluation/datasets/`
- `evaluation/manual_reviews/template.json`
- `evaluation/README.md`
- `requirements.txt`
- `requirements-dev.txt`
- `pytest.ini`
- `tests/` (all hermetic + older scripts; pytest.ini selects hermetic only)

### 3. `docs: README, architecture, evaluation summary, LinkedIn drafts`

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/evaluation_results.md`
- `docs/LINKEDIN.md`
- `docs/screenshots/README.md`
- `docs/GIT_COMMIT_PLAN.md` (this file)

### 4. (Optional) `docs: add UI screenshots`

- Only after you capture PNGs and decide they contain no private document text you do not want public. Remove the `docs/screenshots/*.png` ignore lines if you want images tracked, or force-add carefully.

## Commands (run only when you approve)

```bash
# Review what would be committed
git status -u
git check-ignore -v data/uploads/*.pdf evaluation/results/*.json .env

# Stage in the order above, then for each commit:
git add <paths>
git commit -m "$(cat <<'EOF'
message here

EOF
)"

# Create GitHub repo + push ONLY after explicit approval:
# gh repo create <name> --private --source=. --remote=origin
# git push -u origin main
```

Start **private**, review the GitHub file list, then switch to public when ready.
