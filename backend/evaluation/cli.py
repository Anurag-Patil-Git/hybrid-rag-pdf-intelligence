"""CLI for offline RAG evaluation.

Examples
--------
Retrieve only (recommended on CPU first)::

    python -m backend.evaluation.cli retrieve \\
        --dataset evaluation/datasets/capstone_v1.json \\
        --top-k 5 --limit 5

Full verified set::

    python -m backend.evaluation.cli retrieve \\
        --dataset evaluation/datasets/capstone_v1.json --top-k 5

Optional answer generation (loads Ollama)::

    python -m backend.evaluation.cli answers \\
        --dataset evaluation/datasets/capstone_v1.json --limit 3

List saved runs::

    python -m backend.evaluation.cli list-runs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.core.config import get_settings
from backend.document_registry import DocumentRegistry
from backend.evaluation.dataset import dataset_summary, load_dataset
from backend.evaluation.results import list_runs, load_manual_reviews
from backend.evaluation.runner import (
    load_bundle_from_service,
    run_answer_experiment,
    run_retrieval_experiment,
)
from backend.services.document_service import DocumentService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_chunks_for_sources(vector_store, sources: list[str]):
    """Read existing Chroma rows for BM25. Does not add or delete vectors."""

    from langchain_core.documents import Document

    chunks: list[Document] = []
    for source in sources:
        records = vector_store._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        texts = records.get("documents") or []
        metas = records.get("metadatas") or []
        for text, metadata in zip(texts, metas):
            chunks.append(
                Document(page_content=text or "", metadata=dict(metadata or {}))
            )
    return chunks


def _build_service(settings):
    """Load registry + vectors + BM25 without chat and without reconcile writes."""

    from backend.embedding import get_embedding_model
    from backend.vector_store import create_vector_store

    embedding_model = get_embedding_model()
    vector_store = create_vector_store(
        embedding_model, persist_directory=settings.vector_db_dir
    )

    def set_bm25(_chunks):
        return None

    service = DocumentService(
        registry=DocumentRegistry(db_path=settings.documents_db),
        vector_store=vector_store,
        settings=settings,
        set_bm25_documents=set_bm25,
    )
    # Read-only bootstrap: do not call reconcile(), which can rewrite SQLite
    # timestamps and optional cleanup state.
    indexed = service.registry.list_by_status("indexed")
    for record in indexed:
        source = record["stored_filename"]
        chunks = _load_chunks_for_sources(vector_store, [source])
        service._chunks_by_source[source] = chunks
    return service


def cmd_retrieve(args: argparse.Namespace) -> int:
    settings = get_settings()
    service = _build_service(settings)
    bundle = load_bundle_from_service(
        service,
        load_reranker=True,
    )
    payload = run_retrieval_experiment(
        dataset_path=args.dataset,
        bundle=bundle,
        document_records=service.list_documents(),
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        limit=args.limit,
        results_directory=args.results_dir,
    )
    print(f"Saved run {payload['run_id']} -> {payload.get('saved_to')}")
    for mode, stats in payload["aggregates"].items():
        print(
            f"{mode:16} P@{args.top_k}={_fmt(stats['precision_at_k'])}  "
            f"R@{args.top_k}={_fmt(stats['recall_at_k'])}  "
            f"MRR={_fmt(stats['mrr'])}  "
            f"latency={_fmt(stats['mean_latency_seconds'])}s"
        )
    return 0


def cmd_answers(args: argparse.Namespace) -> int:
    from backend.llm import get_llm

    settings = get_settings()
    service = _build_service(settings)
    bundle = load_bundle_from_service(service, load_reranker=True)
    reviews = load_manual_reviews(args.manual_reviews)
    payload = run_answer_experiment(
        dataset_path=args.dataset,
        bundle=bundle,
        document_records=service.list_documents(),
        llm=get_llm(),
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        retrieval_mode=args.mode,
        limit=args.limit,
        results_directory=args.results_dir,
        manual_reviews=reviews,
    )
    print(f"Saved run {payload['run_id']} -> {payload.get('saved_to')}")
    stats = next(iter(payload["aggregates"].values()))
    print(
        "citation_page_match="
        f"{_fmt(stats.get('mean_citation_page_match'))}  "
        f"answer_latency={_fmt(stats.get('mean_answer_latency_seconds'))}s  "
        f"manual_faithfulness={_fmt(stats.get('manual_faithfulness_mean'))}  "
        f"manual_relevance={_fmt(stats.get('manual_answer_relevance_mean'))}"
    )
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    runs = list_runs(args.results_dir)
    if not runs:
        print("No evaluation runs found.")
        return 0
    for item in runs:
        print(
            f"{item['run_id']}  kind={item['kind']}  "
            f"dataset={item['dataset_version']}  "
            f"questions={item['verified_questions']}  "
            f"top_k={item['top_k']}"
        )
    return 0


def cmd_dataset_info(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    summary = dataset_summary(dataset)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG evaluation")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "evaluation" / "results"),
        help="Directory for JSON run files (never overwrites existing runs)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    retrieve = sub.add_parser("retrieve", help="Compare four retrieval modes")
    retrieve.add_argument("--dataset", required=True)
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--fetch-k", type=int, default=None)
    retrieve.add_argument("--limit", type=int, default=None, help="Score only the first N verified questions")
    retrieve.set_defaults(func=cmd_retrieve)

    answers = sub.add_parser("answers", help="Generate answers for manual review")
    answers.add_argument("--dataset", required=True)
    answers.add_argument("--top-k", type=int, default=5)
    answers.add_argument("--fetch-k", type=int, default=None)
    answers.add_argument("--limit", type=int, default=None)
    answers.add_argument("--mode", default="hybrid_rerank", choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    answers.add_argument("--manual-reviews", default=None, help="Optional JSON with faithfulness/relevance scores")
    answers.set_defaults(func=cmd_answers)

    listing = sub.add_parser("list-runs", help="List saved evaluation JSON files")
    listing.set_defaults(func=cmd_list_runs)

    info = sub.add_parser("dataset-info", help="Show dataset size and verified count")
    info.add_argument("--dataset", required=True)
    info.set_defaults(func=cmd_dataset_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
