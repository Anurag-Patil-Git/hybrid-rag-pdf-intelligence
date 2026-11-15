"""Administrative commands. Does not load any ML model.

    python -m backend.admin report            # read-only, same as 'cleanup'
    python -m backend.admin cleanup           # dry run: shows what would change
    python -m backend.admin cleanup --apply   # moves duplicate files to
                                              # data/quarantine/ and exports then
                                              # removes orphan vectors
"""

from __future__ import annotations

import argparse
import json

from backend.core.config import get_settings
from backend.document_registry import DocumentRegistry
from backend.services.document_service import DocumentService
from backend.vector_store import create_vector_store


def build_service() -> DocumentService:
    settings = get_settings()

    return DocumentService(
        registry=DocumentRegistry(db_path=settings.documents_db),
        vector_store=create_vector_store(None, persist_directory=settings.vector_db_dir),
        settings=settings,
        set_bm25_documents=lambda chunks: None,  # BM25 is rebuilt by the API at startup
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.admin")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="Show duplicate files and orphan vectors (read-only).")
    cleanup = sub.add_parser("cleanup", help="Dry run unless --apply is given.")
    cleanup.add_argument("--apply", action="store_true", help="Actually quarantine/remove.")
    args = parser.parse_args(argv)

    service = build_service()

    if args.command == "cleanup" and args.apply:
        print(json.dumps(service.apply_cleanup(), indent=2))
    else:
        print(json.dumps(service.plan_cleanup(), indent=2))
        if args.command == "cleanup":
            print("\nDry run only. Re-run with --apply to quarantine/remove the items above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
