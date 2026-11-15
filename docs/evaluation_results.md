# Published evaluation summary

Source run: `20261114T121437Z_e1eca06c` (retrieval only, top_k=5, 35 verified questions).

Full per-question JSON files under `evaluation/results/` are **local artifacts**. They contain retrieved chunk text from indexed PDFs and machine-local registry IDs, so they are gitignored and not intended for public commits.

## Dataset

| Field | Value |
|---|---|
| Dataset | `evaluation/datasets/capstone_v1.json` |
| Version | `capstone-v1` |
| Verified questions | 35 |
| Unverified drafts | 1 |
| Documents in scope | LangChain component notes; Non-Native Prompt Engineering Risks |

Labels use stable `doc_key` + page. `document_id` values are resolved at runtime and are not hard-coded in the dataset.

## Automatic retrieval metrics

| Mode | Precision@5 | Recall@5 | MRR | Mean latency (s) |
|---|---|---|---|---|
| dense | 0.2514 | 0.9429 | 0.7295 | 0.0183 |
| bm25 | 0.2457 | 0.9429 | 0.8381 | 0.0004 |
| hybrid (RRF) | 0.2743 | 1.0000 | 0.8857 | 0.0194 |
| hybrid + rerank | 0.2743 | 1.0000 | 0.9095 | 0.6048 |

## Notes

- Precision@5 is chunk-level; page-level labels can score multiple chunks from one page as relevant.
- Recall@5 counts distinct labelled units recovered (page or chunk id), so it stays in `[0, 1]`.
- Citation page match, faithfulness and answer relevance were **not** measured in this retrieval-only run.
- These numbers are from a CPU-only machine with `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. They are not production SLOs.
