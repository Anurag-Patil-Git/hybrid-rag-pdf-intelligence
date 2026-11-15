"""RAG Evaluation dashboard backed by real experiment JSON only."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from frontend.components import empty_state, page_header

RESULTS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "results"
DATASETS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "datasets"


def _load_runs() -> list[tuple[Path, dict]]:
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append((path, payload))
    return runs


def _fmt(value) -> str:
    if value is None:
        return "Not measured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataset_counts() -> dict | None:
    preferred = DATASETS_DIR / "capstone_v1.json"
    path = preferred if preferred.exists() else None
    if path is None:
        candidates = sorted(DATASETS_DIR.glob("*.json"))
        path = candidates[0] if candidates else None
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    questions = payload.get("questions") or []
    verified = sum(1 for item in questions if item.get("verified") is True)
    return {
        "path": path.name,
        "dataset_version": payload.get("dataset_version"),
        "total_questions": len(questions),
        "verified_questions": verified,
    }


page_header(
    "Quality",
    "RAG Evaluation",
    "Measured retrieval and answer results from evaluation/results. No sample scores.",
)

dataset_info = _dataset_counts()
if dataset_info:
    first, second, third = st.columns(3)
    first.metric("Dataset file", dataset_info["path"])
    second.metric("Questions", dataset_info["total_questions"])
    third.metric("Verified", dataset_info["verified_questions"])
    st.caption(f"Dataset version: {dataset_info['dataset_version']}")
else:
    st.info("No evaluation dataset file found under evaluation/datasets/.")

runs = _load_runs()
if not runs:
    empty_state(
        "No evaluation results yet",
        "Run a retrieval experiment, then refresh this page.\n\n"
        "```bash\n"
        "python -m backend.evaluation retrieve \\\n"
        "  --dataset evaluation/datasets/capstone_v1.json --top-k 5 --limit 5\n"
        "```\n\n"
        "Automatic metrics: Precision@K, Recall@K, MRR, latency. "
        "Citation page match is a proxy. Faithfulness and answer relevance stay blank "
        "until a manual review file is provided. Unmeasured metrics stay blank.",
    )
    st.stop()

labels = [f"{payload.get('run_id')} ({payload.get('kind')})" for _, payload in runs]
choice = st.selectbox("Evaluation run", labels, index=0)
selected = runs[labels.index(choice)][1]

st.subheader("Run summary")
meta1, meta2, meta3, meta4 = st.columns(4)
meta1.metric("Kind", selected.get("kind") or "—")
meta2.metric("Top-K", selected.get("top_k") or "—")
meta3.metric("Verified questions", selected.get("verified_questions") or 0)
meta4.metric("Dataset", selected.get("dataset_version") or "—")
st.caption(f"Created {selected.get('created_at')} · run_id `{selected.get('run_id')}`")

aggregates = selected.get("aggregates") or {}
if aggregates:
    st.subheader("Retrieval method comparison")
    rows = []
    for mode, stats in aggregates.items():
        rows.append({
            "mode": mode,
            "precision_at_k": stats.get("precision_at_k"),
            "recall_at_k": stats.get("recall_at_k"),
            "mrr": stats.get("mrr"),
            "mean_latency_seconds": stats.get("mean_latency_seconds")
            or stats.get("mean_retrieval_latency_seconds"),
            "mean_citation_page_match": stats.get("mean_citation_page_match"),
            "manual_faithfulness_mean": stats.get("manual_faithfulness_mean"),
            "manual_answer_relevance_mean": stats.get("manual_answer_relevance_mean"),
        })
    st.dataframe(rows, use_container_width=True)
    chart_rows = [
        {
            "mode": row["mode"],
            "Precision@K": row["precision_at_k"],
            "Recall@K": row["recall_at_k"],
            "MRR": row["mrr"],
        }
        for row in rows
        if row["precision_at_k"] is not None
    ]
    if chart_rows:
        st.caption("Automatic retrieval metrics by mode")
        for metric_name in ("Precision@K", "Recall@K", "MRR"):
            st.write(metric_name)
            st.bar_chart({row["mode"]: row[metric_name] for row in chart_rows})
    else:
        st.caption("No automatic retrieval metrics in this run.")

notes = selected.get("metric_notes") or {}
if notes:
    with st.expander("Metric definitions recorded with this run"):
        for key, value in notes.items():
            st.markdown(f"**{key}:** {value}")

st.subheader("Per-question results")
results = selected.get("results") or {}
mode_names = list(results.keys())
if not mode_names:
    st.info("This run has no per-question rows.")
    st.stop()

mode = st.selectbox("Retrieval mode", mode_names)
rows = results.get(mode) or []
question_ids = [row.get("question_id") for row in rows]
selected_qid = st.selectbox("Question", question_ids)
row = next(item for item in rows if item.get("question_id") == selected_qid)

st.markdown(f"**Question:** {row.get('question')}")
st.caption(row.get("expected_answer_criteria") or "")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision@K", _fmt(row.get("precision_at_k")))
m2.metric("Recall@K", _fmt(row.get("recall_at_k")))
m3.metric("MRR", _fmt(row.get("mrr")))
m4.metric(
    "Latency (s)",
    _fmt(row.get("latency_seconds") or row.get("retrieval_latency_seconds")),
)

st.markdown("**Relevant labels**")
st.json(row.get("relevant_labels") or [])

st.markdown("**Retrieved chunks**")
retrieved = row.get("retrieved") or []
if not retrieved:
    st.caption("No chunks retrieved.")
else:
    for index, chunk in enumerate(retrieved, start=1):
        with st.expander(
            f"#{index} {chunk.get('source')} · page {chunk.get('page')} · chunk {chunk.get('chunk_id')}"
        ):
            st.text(chunk.get("text") or "")

if row.get("answer") is not None:
    st.markdown("**Generated answer**")
    st.write(row.get("answer"))
    st.markdown("**Citations**")
    st.json(row.get("citations") or [])
    st.markdown("**Citation page match (proxy)**")
    st.json(row.get("citation_page_match") or {})
    c1, c2 = st.columns(2)
    c1.metric("Manual faithfulness", _fmt(row.get("manual_faithfulness")))
    c2.metric("Manual answer relevance", _fmt(row.get("manual_answer_relevance")))
    if row.get("reviewer_notes"):
        st.caption(row["reviewer_notes"])
    if row.get("end_to_end_latency_seconds") is not None:
        st.caption(
            f"Answer latency { _fmt(row.get('answer_latency_seconds')) }s · "
            f"End-to-end { _fmt(row.get('end_to_end_latency_seconds')) }s"
        )
