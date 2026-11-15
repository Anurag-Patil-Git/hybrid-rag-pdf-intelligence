"""Hermetic tests for the Phase 5 evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from backend.bm25_retriever import BM25Retriever
from backend.evaluation.dataset import (
    DatasetError,
    chunk_is_relevant,
    load_dataset,
    validate_dataset,
    verified_questions,
)
from backend.evaluation.metrics import (
    average,
    citation_page_match,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from backend.evaluation.retrieval_modes import RetrievalBundle, retrieve
from backend.evaluation.results import list_runs, load_run, save_run
from backend.evaluation.runner import run_retrieval_experiment, score_retrieval
from backend.vector_store import add_documents, create_vector_store


def test_precision_at_k_uses_fixed_denominator():
    assert precision_at_k([True, False, True], k=5) == pytest.approx(2 / 5)
    assert precision_at_k([True, True], k=3) == pytest.approx(2 / 3)
    assert precision_at_k([], k=3) == 0.0


def test_recall_at_k_and_missing_labels():
    assert recall_at_k([True, False, True], total_relevant=4, k=3) == pytest.approx(0.5)
    assert recall_at_k([False, False], total_relevant=0, k=2) is None
    assert recall_at_k([], total_relevant=2, k=5) == 0.0


def test_mrr_conventions():
    assert mean_reciprocal_rank([False, True, True]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([False, False]) == 0.0
    assert mean_reciprocal_rank([]) is None


def test_citation_page_match_is_a_proxy():
    result = citation_page_match([1, 2], [2, 3])
    assert result["score"] == pytest.approx(1 / 3)
    assert "proxy" in result["note"].lower()
    assert citation_page_match([1], [])["score"] is None


def test_average_skips_undefined_values():
    assert average([0.5, None, 1.0]) == pytest.approx(0.75)
    assert average([None, None]) is None


def test_dataset_validation_and_verified_filter(tmp_path):
    payload = {
        "dataset_version": "t1",
        "documents": [
            {
                "doc_key": "doc_a",
                "match_original_filename": "a.pdf",
            }
        ],
        "questions": [
            {
                "question_id": "q1",
                "question": "What is on page 1?",
                "verified": True,
                "expected_answer_criteria": "Mentions alpha.",
                "relevant": [{"doc_key": "doc_a", "page": 1, "chunk_ids": None}],
            },
            {
                "question_id": "q2",
                "question": "Draft",
                "verified": False,
                "expected_answer_criteria": "n/a",
                "relevant": [],
            },
        ],
    }
    path = tmp_path / "set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = load_dataset(path)
    assert len(verified_questions(dataset)) == 1

    with pytest.raises(DatasetError):
        validate_dataset({**payload, "questions": [{
            "question_id": "bad",
            "question": "x",
            "verified": True,
            "expected_answer_criteria": "y",
            "relevant": [],
        }]})


def test_chunk_relevance_by_page_and_chunk_id():
    labels = [{"doc_key": "a", "page": 2, "chunk_ids": None}]
    mapping = {"a": "file.pdf"}
    assert chunk_is_relevant({"source": "file.pdf", "page": 2, "chunk_id": 9}, labels, mapping)
    assert not chunk_is_relevant({"source": "file.pdf", "page": 1, "chunk_id": 9}, labels, mapping)

    chunk_labels = [{"doc_key": "a", "page": 2, "chunk_ids": [4, 5]}]
    assert chunk_is_relevant({"source": "file.pdf", "page": 2, "chunk_id": 4}, chunk_labels, mapping)
    assert not chunk_is_relevant({"source": "file.pdf", "page": 2, "chunk_id": 7}, chunk_labels, mapping)


def test_score_retrieval_with_no_labels():
    docs = [Document(page_content="x", metadata={"source": "a.pdf", "page": 1})]
    scored = score_retrieval(docs, [], {}, top_k=5)
    assert scored["precision_at_k"] is None
    assert scored["recall_at_k"] is None
    assert scored["mrr"] is None


def test_recall_caps_at_one_when_many_chunks_share_a_page():
    docs = [
        Document(page_content="a", metadata={"source": "notes.pdf", "page": 1, "chunk_id": 0}),
        Document(page_content="b", metadata={"source": "notes.pdf", "page": 1, "chunk_id": 1}),
        Document(page_content="c", metadata={"source": "notes.pdf", "page": 1, "chunk_id": 2}),
    ]
    scored = score_retrieval(
        docs,
        [{"doc_key": "notes", "page": 1, "chunk_ids": None}],
        {"notes": "notes.pdf"},
        top_k=5,
    )
    assert scored["precision_at_k"] == pytest.approx(3 / 5)
    assert scored["recall_at_k"] == pytest.approx(1.0)
    assert scored["matched_relevant"] == 1


def _bundle(tmp_path):
    embedding = DeterministicFakeEmbedding(size=16)
    store = create_vector_store(embedding, persist_directory=tmp_path / "vectors")
    chunks = [
        Document(
            page_content="LangChain indexes help retrieve documents efficiently.",
            metadata={"source": "notes.pdf", "page": 1, "chunk_id": 0},
        ),
        Document(
            page_content="Agents can use tools and plan multi-step workflows.",
            metadata={"source": "notes.pdf", "page": 2, "chunk_id": 1},
        ),
        Document(
            page_content="Unrelated cooking recipe with flour and sugar.",
            metadata={"source": "other.pdf", "page": 1, "chunk_id": 0},
        ),
    ]
    add_documents(store, chunks)
    bm25 = BM25Retriever(documents=chunks)
    return RetrievalBundle(vector_store=store, bm25_retriever=bm25, reranker=None), chunks


def test_retrieval_modes_respect_filters_and_top_k(tmp_path):
    bundle, _chunks = _bundle(tmp_path)
    filters = {"source": "notes.pdf"}

    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        docs, latency = retrieve(
            bundle, mode, "indexes retrieve documents", top_k=2, filters=filters
        )
        assert latency >= 0
        assert len(docs) <= 2
        assert all(doc.metadata["source"] == "notes.pdf" for doc in docs)


def test_retrieval_experiment_comparison_consistency(tmp_path):
    bundle, _chunks = _bundle(tmp_path)
    dataset = {
        "dataset_version": "synth-1",
        "documents": [
            {
                "doc_key": "notes",
                "match_original_filename": "notes.pdf",
                "match_stored_filename_contains": "notes.pdf",
            }
        ],
        "questions": [
            {
                "question_id": "q1",
                "question": "How do indexes help retrieve documents?",
                "verified": True,
                "expected_answer_criteria": "Mentions retrieve/indexes.",
                "relevant": [{"doc_key": "notes", "page": 1, "chunk_ids": [0]}],
            }
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    records = [
        {
            "document_id": "d1",
            "original_filename": "notes.pdf",
            "stored_filename": "notes.pdf",
            "status": "indexed",
        }
    ]
    results_dir = tmp_path / "results"
    payload = run_retrieval_experiment(
        dataset_path=dataset_path,
        bundle=bundle,
        document_records=records,
        top_k=2,
        modes=("dense", "bm25", "hybrid", "hybrid_rerank"),
        results_directory=results_dir,
        save=True,
    )

    assert payload["kind"] == "retrieval"
    assert set(payload["results"]) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    # Same question set for every mode
    for mode, rows in payload["results"].items():
        assert [row["question_id"] for row in rows] == ["q1"]
        assert rows[0]["labelled_relevant"] == 1
    saved = list_runs(results_dir)
    assert len(saved) == 1
    reloaded = load_run(saved[0]["path"])
    assert reloaded["run_id"] == payload["run_id"]


def test_save_run_never_overwrites(tmp_path):
    path = save_run({"run_id": "fixed", "kind": "retrieval"}, tmp_path)
    assert path.exists()
    with pytest.raises(FileExistsError):
        save_run({"run_id": "fixed", "kind": "retrieval"}, tmp_path)


def test_evaluation_does_not_call_chat_pipeline(tmp_path, monkeypatch):
    bundle, _chunks = _bundle(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({
            "dataset_version": "synth-1",
            "documents": [{
                "doc_key": "notes",
                "match_original_filename": "notes.pdf",
            }],
            "questions": [{
                "question_id": "q1",
                "question": "indexes retrieve",
                "verified": True,
                "expected_answer_criteria": "indexes",
                "relevant": [{"doc_key": "notes", "page": 1, "chunk_ids": None}],
            }],
        }),
        encoding="utf-8",
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("chat pipeline must not be used")

    monkeypatch.setattr("backend.rag_pipeline.RAGPipeline.ask", boom)

    payload = run_retrieval_experiment(
        dataset_path=dataset_path,
        bundle=bundle,
        document_records=[{
            "document_id": "d1",
            "original_filename": "notes.pdf",
            "stored_filename": "notes.pdf",
        }],
        top_k=1,
        modes=("bm25",),
        results_directory=tmp_path / "out",
        save=False,
    )
    assert payload["results"]["bm25"][0]["question_id"] == "q1"


def test_capstone_dataset_file_validates():
    path = Path("evaluation/datasets/capstone_v1.json")
    if not path.exists():
        pytest.skip("capstone dataset not present")
    dataset = load_dataset(path)
    assert dataset["dataset_version"] == "capstone-v1"
    assert len(verified_questions(dataset)) >= 30
