"""Run retrieval (and optional answer) experiments offline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.documents import Document

from backend.bm25_retriever import BM25Retriever
from backend.context_builder import build_context
from backend.evaluation.dataset import (
    DatasetError,
    chunk_is_relevant,
    labelled_relevant_keys,
    load_dataset,
    resolve_doc_key,
    retrieved_key,
    verified_questions,
)
from backend.evaluation.metrics import (
    average,
    citation_page_match,
    mean_reciprocal_rank,
    precision_at_k,
)
from backend.evaluation.retrieval_modes import (
    RETRIEVAL_MODES,
    RetrievalBundle,
    retrieve,
    serialize_documents,
)
from backend.evaluation.results import new_run_id, save_run, utc_now
from backend.llm import generate_answer

logger = logging.getLogger(__name__)


def build_source_filter(stored_filenames: list[str]) -> dict:
    if len(stored_filenames) == 1:
        return {"source": stored_filenames[0]}
    return {"source": {"$in": list(stored_filenames)}}


def score_retrieval(
    documents: list[Document],
    relevant_items: list[dict],
    doc_key_to_source: dict[str, str],
    top_k: int,
) -> dict:
    """Score one query against labels. Shared by every retrieval mode."""

    label_keys = labelled_relevant_keys(relevant_items, doc_key_to_source)
    if not label_keys:
        return {
            "labelled_relevant": 0,
            "relevance_flags": [],
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
            "note": "No relevant labels; retrieval metrics not measured.",
        }

    flags = []
    matched_labels: set[tuple] = set()
    for document in documents[:top_k]:
        is_relevant = chunk_is_relevant(
            document.metadata, relevant_items, doc_key_to_source
        )
        flags.append(is_relevant)
        if is_relevant:
            key = retrieved_key(document.metadata, relevant_items)
            if key in label_keys:
                matched_labels.add(key)

    # Precision is chunk-level (hits / K). Recall is label-set coverage so
    # several chunks from one labelled page cannot push Recall above 1.0.
    return {
        "labelled_relevant": len(label_keys),
        "matched_relevant": len(matched_labels),
        "relevance_flags": flags,
        "precision_at_k": precision_at_k(flags, top_k),
        "recall_at_k": len(matched_labels) / len(label_keys),
        "mrr": mean_reciprocal_rank(flags),
        "note": None,
    }


def run_retrieval_experiment(
    *,
    dataset_path: str | Path,
    bundle: RetrievalBundle,
    document_records: list[dict],
    top_k: int = 5,
    fetch_k: int | None = None,
    modes: tuple[str, ...] = RETRIEVAL_MODES,
    limit: int | None = None,
    results_directory: str | Path | None = None,
    save: bool = True,
) -> dict:
    """Evaluate retrieval only. Does not call the LLM or touch chat history."""

    dataset = load_dataset(dataset_path)
    questions = verified_questions(dataset)
    if limit is not None:
        questions = questions[: max(0, limit)]
    if not questions:
        raise DatasetError("No verified questions to evaluate.")

    resolved = resolve_doc_key(document_records, dataset["documents"])
    doc_key_to_source = {
        key: record["stored_filename"] for key, record in resolved.items()
    }
    filters = build_source_filter(list(doc_key_to_source.values()))

    per_mode: dict[str, list[dict]] = {mode: [] for mode in modes}

    for question in questions:
        for mode in modes:
            documents, latency = retrieve(
                bundle,
                mode,
                question["question"],
                top_k=top_k,
                fetch_k=fetch_k,
                filters=filters,
            )
            metrics = score_retrieval(
                documents,
                question["relevant"],
                doc_key_to_source,
                top_k,
            )
            per_mode[mode].append({
                "question_id": question["question_id"],
                "question": question["question"],
                "expected_answer_criteria": question["expected_answer_criteria"],
                "relevant_labels": question["relevant"],
                "retrieved": serialize_documents(documents),
                "latency_seconds": latency,
                **metrics,
            })

    aggregates = {
        mode: {
            "precision_at_k": average(
                item["precision_at_k"] for item in rows
            ),
            "recall_at_k": average(item["recall_at_k"] for item in rows),
            "mrr": average(item["mrr"] for item in rows),
            "mean_latency_seconds": average(
                item["latency_seconds"] for item in rows
            ),
            "questions_scored": sum(
                1 for item in rows if item["precision_at_k"] is not None
            ),
        }
        for mode, rows in per_mode.items()
    }

    payload = {
        "run_id": new_run_id(),
        "kind": "retrieval",
        "created_at": utc_now(),
        "dataset_version": dataset["dataset_version"],
        "dataset_path": str(dataset_path),
        "verified_questions": len(questions),
        "top_k": top_k,
        "fetch_k": fetch_k or max(top_k * 3, top_k),
        "modes": list(modes),
        "document_scope": {
            key: {
                "document_id": record["document_id"],
                "original_filename": record["original_filename"],
                "stored_filename": record["stored_filename"],
            }
            for key, record in resolved.items()
        },
        "aggregates": aggregates,
        "results": per_mode,
        "metric_notes": {
            "precision_at_k": "hits / K; K is always the configured cutoff",
            "recall_at_k": "hits / labelled relevant units; None when unlabelled",
            "mrr": "1/rank of first hit; 0 when labelled but no hit; None when unlabelled",
            "citation_page_match": "not computed in retrieval-only runs",
        },
    }

    if save:
        path = save_run(payload, results_directory)
        payload["saved_to"] = str(path)
        logger.info("Saved retrieval evaluation to %s", path)

    return payload


def run_answer_experiment(
    *,
    dataset_path: str | Path,
    bundle: RetrievalBundle,
    document_records: list[dict],
    llm,
    top_k: int = 5,
    fetch_k: int | None = None,
    retrieval_mode: str = "hybrid_rerank",
    limit: int | None = None,
    results_directory: str | Path | None = None,
    manual_reviews: dict | None = None,
    save: bool = True,
) -> dict:
    """Generate answers for inspection. Faithfulness/relevance stay manual."""

    dataset = load_dataset(dataset_path)
    questions = verified_questions(dataset)
    if limit is not None:
        questions = questions[: max(0, limit)]
    if not questions:
        raise DatasetError("No verified questions to evaluate.")

    resolved = resolve_doc_key(document_records, dataset["documents"])
    doc_key_to_source = {
        key: record["stored_filename"] for key, record in resolved.items()
    }
    filters = build_source_filter(list(doc_key_to_source.values()))
    reviews = manual_reviews or {}

    rows = []
    for question in questions:
        retrieval_started = time.perf_counter()
        documents, retrieval_latency = retrieve(
            bundle,
            retrieval_mode,
            question["question"],
            top_k=top_k,
            fetch_k=fetch_k,
            filters=filters,
        )
        context, citations = build_context(documents)
        answer_started = time.perf_counter()
        answer = generate_answer(
            question=question["question"],
            context=context,
            llm=llm,
        )
        answer_latency = time.perf_counter() - answer_started
        end_to_end = time.perf_counter() - retrieval_started

        labelled_pages = [item["page"] for item in question["relevant"]]
        cited_pages = [item.get("page") for item in citations]
        citation_proxy = citation_page_match(cited_pages, labelled_pages)
        review = reviews.get(question["question_id"], {})

        rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "expected_answer_criteria": question["expected_answer_criteria"],
            "relevant_labels": question["relevant"],
            "retrieved": serialize_documents(documents),
            "context": context,
            "answer": answer,
            "citations": citations,
            "retrieval_latency_seconds": retrieval_latency,
            "answer_latency_seconds": answer_latency,
            "end_to_end_latency_seconds": end_to_end,
            "citation_page_match": citation_proxy,
            "manual_faithfulness": review.get("faithfulness"),
            "manual_answer_relevance": review.get("answer_relevance"),
            "reviewer_notes": review.get("notes"),
        })

    payload = {
        "run_id": new_run_id(),
        "kind": "answers",
        "created_at": utc_now(),
        "dataset_version": dataset["dataset_version"],
        "dataset_path": str(dataset_path),
        "verified_questions": len(questions),
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "aggregates": {
            retrieval_mode: {
                "mean_retrieval_latency_seconds": average(
                    row["retrieval_latency_seconds"] for row in rows
                ),
                "mean_answer_latency_seconds": average(
                    row["answer_latency_seconds"] for row in rows
                ),
                "mean_end_to_end_latency_seconds": average(
                    row["end_to_end_latency_seconds"] for row in rows
                ),
                "mean_citation_page_match": average(
                    (row["citation_page_match"] or {}).get("score") for row in rows
                ),
                "manual_faithfulness_mean": average(
                    row["manual_faithfulness"] for row in rows
                ),
                "manual_answer_relevance_mean": average(
                    row["manual_answer_relevance"] for row in rows
                ),
            }
        },
        "results": {retrieval_mode: rows},
        "metric_notes": {
            "citation_page_match": (
                "Page-overlap proxy only; not proof of factual support."
            ),
            "faithfulness": "Manual reviewer score when provided; otherwise null.",
            "answer_relevance": "Manual reviewer score when provided; otherwise null.",
        },
    }

    if save:
        path = save_run(payload, results_directory)
        payload["saved_to"] = str(path)
        logger.info("Saved answer evaluation to %s", path)

    return payload


def load_bundle_from_service(
    document_service,
    *,
    load_reranker: bool = True,
    load_embeddings: bool = False,
) -> RetrievalBundle:
    """Build a RetrievalBundle from an existing DocumentService state."""

    from backend.reranker import get_reranker

    chunks = [
        chunk
        for chunk_list in document_service._chunks_by_source.values()
        for chunk in chunk_list
    ]
    bm25 = BM25Retriever(documents=chunks) if chunks else None
    reranker = get_reranker() if load_reranker else None
    return RetrievalBundle(
        vector_store=document_service.vector_store,
        bm25_retriever=bm25,
        reranker=reranker,
        embedding_model=None,
    )
