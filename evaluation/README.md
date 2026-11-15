# RAG Evaluation

Offline evaluation for retrieval quality. This path does **not** use the `/chat` API and does not write conversation history.

## Dataset format

Files live in `evaluation/datasets/*.json`.

| Field | Meaning |
|---|---|
| `dataset_version` | Version string recorded in every run |
| `documents[].doc_key` | Stable label key used by questions |
| `documents[].match_original_filename` | Match registry `original_filename` (exact or substring) |
| `documents[].match_stored_filename_contains` | Match registry `stored_filename` substring |
| `questions[].verified` | `true` = scored; `false` = ignored until labelled |
| `questions[].relevant` | List of `{doc_key, page, chunk_ids}` |

**Do not hard-code `document_id`.** IDs change across machines. Matchers resolve to the current registry at run time.

**Chunk IDs are optional.** When `chunk_ids` is `null`, a retrieved chunk is relevant if its `source` and `page` match a label. When `chunk_ids` is a list, only those chunk ids count.

### How to add questions

1. Inspect the PDF pages yourself.
2. Copy `template.json` or edit `capstone_v1.json`.
3. Set `verified: true` only after you confirm the relevant page(s).
4. Leave `verified: false` for drafts.
5. Run `python -m backend.evaluation dataset-info --dataset ...` to confirm counts.

`capstone_v1.json` contains **35 verified** questions labelled from inspected local PDFs (LangChain notes + Non-Native Prompt Engineering Risks), plus one unverified draft.

## Metrics

| Metric | Definition |
|---|---|
| Precision@K | Relevant hits in top-K / K (always divides by K) |
| Recall@K | Distinct labelled units recovered in top-K / labelled units (`null` if unlabelled; capped at 1.0) |
| MRR | 1 / rank of first relevant hit; `0` if labelled but missed; `null` if unlabelled |
| Citation page match | Jaccard overlap of cited pages vs labelled pages (**proxy only**) |
| Faithfulness / answer relevance | Manual reviewer fields only |

## Commands

From the project root, with the same environment as the API:

```bash
# Dataset summary
python -m backend.evaluation dataset-info --dataset evaluation/datasets/capstone_v1.json

# Small CPU smoke run (first 5 verified questions)
python -m backend.evaluation retrieve \
  --dataset evaluation/datasets/capstone_v1.json \
  --top-k 5 --limit 5

# Full verified retrieval comparison
python -m backend.evaluation retrieve \
  --dataset evaluation/datasets/capstone_v1.json \
  --top-k 5

# Optional answer generation for manual review (needs Ollama)
python -m backend.evaluation answers \
  --dataset evaluation/datasets/capstone_v1.json \
  --limit 3 \
  --manual-reviews evaluation/manual_reviews/template.json

# List saved runs
python -m backend.evaluation list-runs
```

Results are written under `evaluation/results/<run_id>.json`. Existing run files are never overwritten.

The Streamlit **RAG Evaluation** page reads the newest run from that folder.

## Isolation

* Evaluation loads Chroma + BM25 for search only.
* It does not call `RAGPipeline.ask`.
* It does not call document `reconcile()`, so it will not rewrite the document registry.
* It does not modify uploads or conversation history.
* Opening Chroma may rewrite `vector_db/chroma.sqlite3` internal pages even for read-only queries. PDFs and conversation history stay untouched.
* Prefer `--limit` on CPU before a full run. Models are loaded once per process.
