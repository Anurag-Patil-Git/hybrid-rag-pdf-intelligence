"""Document lifecycle management.

SQLite (``DocumentRegistry``) is the source of truth for which documents exist.
This service keeps the PDF files, the ChromaDB vectors and the in-memory BM25
index consistent with that registry.

Every multi-step operation is ordered so that a crash leaves something the
startup ``reconcile()`` can finish or clean up:

* upload  - file -> registry('processing') -> vectors -> registry('indexed');
            any failure rolls everything back.
* delete  - registry('deleting') -> vectors -> file -> registry row removed;
            a crash leaves 'deleting', which reconcile completes.
* replace - new file + new vectors are built first; the registry record is
            switched to them in one SQL statement; the old file and vectors
            are removed last. The old document stays searchable until then.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Callable
from uuid import uuid4

from langchain_core.documents import Document

from backend.chunking import split_documents
from backend.core.config import Settings
from backend.document_processor import extract_pdf_pages
from backend.document_registry import DocumentRegistry, DuplicateDocumentError
from backend.vector_store import add_documents, count_chunks_by_source, delete_source

logger = logging.getLogger(__name__)

_UUID_PREFIX = re.compile(r"^[0-9a-f]{32}_", re.IGNORECASE)
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_STEM_LENGTH = 120


# --------------------------------------------------
# Errors (mapped to HTTP status codes in backend/main.py)
# --------------------------------------------------

class DocumentError(Exception):
    """Base class for document lifecycle errors."""


class InvalidUpload(DocumentError):
    """The uploaded file was rejected. ``status_code`` suggests the HTTP code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class DuplicateDocument(DocumentError):
    """The same bytes (SHA-256) are already registered."""

    def __init__(self, existing_document: dict):
        super().__init__("This PDF has already been uploaded.")
        self.existing_document = existing_document


class DocumentNotFound(DocumentError):
    """No registry record with that ID."""


class DocumentUnavailable(DocumentError):
    """Chat was requested against documents that cannot be searched."""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


class InvalidSearch(DocumentError):
    """The chat request named documents in a way that cannot be searched."""

    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 422


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file without loading it fully into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def clean_display_name(filename: str | None) -> str:
    """Return the bare filename of an upload, or raise InvalidUpload."""

    if not filename or not filename.strip():
        raise InvalidUpload("Filename is required.", 400)

    name = Path(filename.replace("\\", "/")).name.strip()

    if Path(name).suffix.lower() != ".pdf":
        raise InvalidUpload("Only PDF files are supported.", 400)

    return name


def _make_stored_name(display_name: str) -> str:
    """Unique, filesystem-safe name (safe on Windows and Linux)."""

    path = Path(display_name)
    stem = _UNSAFE_FILENAME_CHARS.sub("_", path.stem)[:_MAX_STEM_LENGTH] or "document"

    return f"{uuid4().hex}_{stem}.pdf"


def _legacy_display_name(stored_name: str) -> str:
    return _UUID_PREFIX.sub("", stored_name)


# --------------------------------------------------
# Service
# --------------------------------------------------

class DocumentService:
    """Owns the document lifecycle across registry, files, ChromaDB and BM25."""

    def __init__(
        self,
        registry: DocumentRegistry,
        vector_store,
        settings: Settings,
        set_bm25_documents: Callable[[list[Document]], None],
    ):
        self.registry = registry
        self.vector_store = vector_store
        self.settings = settings
        self._set_bm25_documents = set_bm25_documents

        self._lock = RLock()
        self._chunks_by_source: dict[str, list[Document]] = {}
        self.last_reconcile_report: dict | None = None

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- validation ----------------

    def validate_upload(self, filename: str | None, data: bytes) -> str:
        """Check an upload; return the cleaned display filename."""

        name = clean_display_name(filename)

        if not data:
            raise InvalidUpload("Uploaded file is empty.", 400)

        if len(data) > self.settings.max_upload_bytes:
            raise InvalidUpload(
                f"File exceeds the {self.settings.max_upload_mb} MB upload limit.",
                413,
            )

        if not data.startswith(b"%PDF-"):
            raise InvalidUpload(
                "The uploaded file does not appear to be a valid PDF.", 400
            )

        return name

    # ---------------- queries ----------------

    def list_documents(self) -> list[dict]:
        return self.registry.list_documents()

    def get_document(self, document_id: str) -> dict:
        record = self.registry.get_by_id(document_id)

        if record is None:
            raise DocumentNotFound(f"Document {document_id} not found.")

        return record

    def stats(self) -> dict:
        indexed = self.registry.list_by_status("indexed")

        return {
            "indexed_documents": len(indexed),
            "total_documents": len(self.registry.list_documents()),
            "total_pages": sum(item["page_count"] for item in indexed),
            "total_chunks": sum(item["chunk_count"] for item in indexed),
        }

    def retrieval_filters(
        self,
        source: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        """Metadata filter for retrieval. See :meth:`resolve_search`."""

        filters, _labels = self.resolve_search(source, document_ids)
        return filters

    def resolve_search(
        self,
        source: str | None = None,
        document_ids: list[str] | None = None,
    ) -> tuple[dict, dict]:
        """
        Decide which indexed documents a chat request may search.

        ``document_ids`` is the primary selector. ``source`` (a stored
        filename) remains for older clients. Omitting both searches every
        indexed document. An empty ``document_ids`` list is a deliberate
        empty selection and is rejected. Supplying both is allowed only
        when they name the same single document.

        Returns the Chroma/BM25 metadata filter and a label map keyed by
        stored filename. Every retrieval stage uses the filter. Citations
        are limited to the label map, so a chunk outside the selection
        cannot be shown.
        """

        indexed = self.registry.list_by_status("indexed")
        indexed_by_id = {item["document_id"]: item for item in indexed}
        indexed_by_source = {item["stored_filename"]: item for item in indexed}

        if document_ids is None and not indexed:
            raise DocumentUnavailable("Upload a PDF before asking questions.", 409)

        selected = self._records_for_ids(document_ids, indexed_by_id)

        if document_ids is not None and not selected:
            if source:
                raise InvalidSearch(
                    "source and document_ids select different documents. "
                    "Pass only one of them, or make them select the same document."
                )
            raise InvalidSearch(
                "Select at least one document, or omit document_ids to search "
                "every indexed document."
            )

        source_record = None
        if source is not None:
            if not indexed:
                raise DocumentUnavailable("Upload a PDF before asking questions.", 409)
            source_record = indexed_by_source.get(source)
            if source_record is None:
                raise DocumentUnavailable(
                    "The selected document is not available. "
                    "It may have been deleted or is not indexed.",
                    404,
                )

        if selected is not None and source_record is not None:
            selected_ids = {item["document_id"] for item in selected}
            if selected_ids != {source_record["document_id"]}:
                raise InvalidSearch(
                    "source and document_ids select different documents. "
                    "Pass only one of them, or make them select the same document."
                )

        if selected is None and source_record is None:
            chosen = indexed
            many = True
        elif source_record is not None:
            chosen = [source_record]
            many = False
        else:
            chosen = selected
            many = len(chosen) != 1

        stored_names = [item["stored_filename"] for item in chosen]
        filters = (
            {"source": {"$in": stored_names}}
            if many
            else {"source": stored_names[0]}
        )
        labels = {
            item["stored_filename"]: {
                "document_id": item["document_id"],
                "original_filename": item["original_filename"],
            }
            for item in chosen
        }
        return filters, labels

    def _records_for_ids(
        self,
        document_ids: list[str] | None,
        indexed_by_id: dict,
    ) -> list[dict] | None:
        """Resolve document IDs. ``None`` means the caller did not choose any."""

        if document_ids is None:
            return None

        selected = []
        seen = set()

        for value in document_ids:
            if not isinstance(value, str) or not value.strip():
                raise InvalidSearch("Each document ID must be a non-empty string.")

            document_id = value.strip()
            if document_id in seen:
                continue
            seen.add(document_id)

            record = indexed_by_id.get(document_id)
            if record is None:
                raise DocumentUnavailable(
                    "One or more selected documents are not available. "
                    "They may have been deleted or are not indexed.",
                    404,
                )
            selected.append(record)

        return selected

    # ---------------- upload ----------------

    def upload(self, filename: str | None, data: bytes) -> dict:
        """Validate, de-duplicate, store and index a new PDF."""

        display_name = self.validate_upload(filename, data)
        file_hash = sha256_bytes(data)

        with self._lock:
            existing = self.registry.get_by_hash(file_hash)

            if existing is not None:
                if existing["status"] != "failed":
                    raise DuplicateDocument(existing)

                # Same bytes failed earlier (e.g. interrupted). Clear the
                # leftovers of that attempt and try again from scratch.
                logger.info("Re-uploading previously failed document %s", existing["document_id"])
                self._remove_everywhere(existing)

            return self._store_and_index(display_name, data, file_hash)

    def _store_and_index(self, display_name: str, data: bytes, file_hash: str) -> dict:
        stored_name = _make_stored_name(display_name)
        final_path = self.settings.upload_dir / stored_name
        temp_path = final_path.with_name(stored_name + ".part")
        document_id = str(uuid4())
        registered = False

        try:
            # .part files are ignored by the *.pdf scan, so an interrupted
            # write can never be mistaken for a real document.
            temp_path.write_bytes(data)
            os.replace(temp_path, final_path)

            try:
                self.registry.create_document(
                    document_id=document_id,
                    original_filename=display_name,
                    stored_filename=stored_name,
                    file_hash=file_hash,
                    file_size=len(data),
                )
            except DuplicateDocumentError as error:
                raise DuplicateDocument(error.existing_document) from error

            registered = True

            pages, chunks = self._extract_and_chunk(final_path)

            add_documents(self.vector_store, chunks)
            self._chunks_by_source[stored_name] = chunks
            self._rebuild_bm25()

            self.registry.update_status(
                document_id,
                status="indexed",
                page_count=pages,
                chunk_count=len(chunks),
            )

        except DuplicateDocument:
            final_path.unlink(missing_ok=True)
            raise

        except Exception as error:
            logger.exception("Indexing failed for %s; rolling back", display_name)
            self._rollback_new_document(stored_name, document_id if registered else None)
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)

            if isinstance(error, InvalidUpload):
                raise

            raise InvalidUpload(
                f"The PDF could not be processed: {error}", 422
            ) from error

        logger.info(
            "Indexed %s (%d pages, %d chunks)", display_name, pages, len(chunks)
        )

        return self.registry.get_by_id(document_id)

    def _extract_and_chunk(self, path: Path) -> tuple[int, list[Document]]:
        pages = extract_pdf_pages(path)

        chunks = split_documents(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        if not chunks:
            raise InvalidUpload("No text chunks could be generated.", 422)

        return len(pages), chunks

    def _rollback_new_document(self, stored_name: str, document_id: str | None) -> None:
        """Undo a failed upload. Never raises; problems are logged."""

        try:
            delete_source(self.vector_store, stored_name)
        except Exception:
            logger.exception("Rollback: could not delete vectors for %s", stored_name)

        if self._chunks_by_source.pop(stored_name, None) is not None:
            self._rebuild_bm25()

        if document_id is not None:
            try:
                self.registry.delete_document(document_id)
            except Exception:
                logger.exception("Rollback: could not delete registry record %s", document_id)

    # ---------------- delete ----------------

    def delete(self, document_id: str) -> dict:
        """Remove a document from registry, disk, ChromaDB and BM25."""

        with self._lock:
            record = self.get_document(document_id)

            # Marking first means a crash leaves 'deleting', which reconcile
            # finishes. Searchable immediately stops: only 'indexed' is searched.
            self.registry.update_status(document_id, status="deleting")
            self._remove_everywhere(record)

        logger.info("Deleted document %s (%s)", document_id, record["original_filename"])

        return record

    def _remove_everywhere(self, record: dict) -> None:
        """Idempotent removal of one document's vectors, file and registry row."""

        stored_name = record["stored_filename"]

        removed = delete_source(self.vector_store, stored_name)

        if self._chunks_by_source.pop(stored_name, None) is not None:
            self._rebuild_bm25()

        (self.settings.upload_dir / stored_name).unlink(missing_ok=True)
        self.registry.delete_document(record["document_id"])

        logger.debug("Removed %d vectors for %s", removed, stored_name)

    # ---------------- replace ----------------

    def replace(self, document_id: str, filename: str | None, data: bytes) -> dict:
        """
        Replace the PDF behind an existing document.

        Journalled in the registry's ``operations`` table:

        1. journal 'indexing'   - registry still points at the OLD version
        2. build the new file + vectors (old version stays searchable)
        3. one SQL transaction  - registry -> new version AND journal ->
                                  'cleanup_pending'
        4. remove old vectors and file, then drop the journal entry

        A crash before step 3 leaves the old version active (recovery removes
        the half-built new one). A crash after step 3 leaves the new version
        active (recovery finishes the cleanup). Returns
        ``{"document": record, "changed": bool}``.
        """

        display_name = self.validate_upload(filename, data)
        new_hash = sha256_bytes(data)

        with self._lock:
            record = self.get_document(document_id)

            if record["status"] == "deleting":
                raise InvalidUpload("This document is being deleted.", 409)

            if record["status"] != "indexed":
                raise InvalidUpload(
                    f"Only indexed documents can be replaced (status: {record['status']}).",
                    409,
                )

            if new_hash == record["file_hash"]:
                return {"document": record, "changed": False}

            other = self.registry.get_by_hash(new_hash)

            if other is not None:
                raise DuplicateDocument(other)

            if any(
                op["document_id"] == document_id
                for op in self.registry.list_operations()
            ):
                raise InvalidUpload(
                    "A previous replacement of this document is still being "
                    "cleaned up. Restart the service or retry shortly.",
                    409,
                )

            old_stored = record["stored_filename"]
            new_stored = _make_stored_name(display_name)
            new_path = self.settings.upload_dir / new_stored
            temp_path = new_path.with_name(new_stored + ".part")
            operation_id = str(uuid4())

            self.registry.begin_replace_operation(
                operation_id, document_id, old_stored, new_stored
            )

            try:
                temp_path.write_bytes(data)
                os.replace(temp_path, new_path)

                pages, chunks = self._extract_and_chunk(new_path)
                add_documents(self.vector_store, chunks)

                updated = self.registry.replace_file(
                    document_id=document_id,
                    original_filename=display_name,
                    stored_filename=new_stored,
                    file_hash=new_hash,
                    file_size=len(data),
                    page_count=pages,
                    chunk_count=len(chunks),
                    operation_id=operation_id,
                )

                if updated is None:
                    raise DocumentNotFound(f"Document {document_id} not found.")

            except Exception as error:
                logger.exception("Replace failed for %s; old version kept", document_id)
                self._discard_new_version(new_stored)
                temp_path.unlink(missing_ok=True)
                self.registry.finish_operation(operation_id)

                if isinstance(error, (DocumentError, sqlite3.IntegrityError)):
                    raise

                raise InvalidUpload(
                    f"The PDF could not be processed: {error}", 422
                ) from error

            # The registry now points at the new version.
            self._chunks_by_source.pop(old_stored, None)
            self._chunks_by_source[new_stored] = chunks
            self._rebuild_bm25()

            try:
                self._finish_replace_cleanup(
                    {
                        "operation_id": operation_id,
                        "old_stored_filename": old_stored,
                    }
                )
            except Exception:
                # Journal entry stays; reconcile() finishes it on next start.
                logger.exception(
                    "Old-version cleanup for %s is pending", old_stored
                )

        return {"document": updated, "changed": True}

    def _discard_new_version(self, new_stored: str) -> None:
        """Best-effort removal of a never-activated new version."""

        try:
            delete_source(self.vector_store, new_stored)
        except Exception:
            logger.exception("Could not delete vectors for %s", new_stored)

        try:
            (self.settings.upload_dir / new_stored).unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not delete file %s", new_stored)

    def _finish_replace_cleanup(self, operation: dict) -> None:
        """
        Remove the old version's vectors and file, then the journal entry.

        Idempotent. Raises on failure so the journal entry is kept. Never
        touches a name the registry still references.
        """

        old_stored = operation["old_stored_filename"]

        if self.registry.get_by_source(old_stored) is None:
            delete_source(self.vector_store, old_stored)
            (self.settings.upload_dir / old_stored).unlink(missing_ok=True)
        else:
            logger.warning(
                "Old version %s is referenced by the registry; not removing it",
                old_stored,
            )

        self.registry.finish_operation(operation["operation_id"])

    # ---------------- reconcile ----------------

    def reconcile(self) -> dict:
        """
        Make registry, files, vectors and the BM25 index agree at startup.

        Nothing is deleted unless it is the unambiguous leftover of a
        half-finished operation of this service:

        * 'deleting' records are finished (the user asked for the deletion);
        * leftover ``*.part`` files are removed (incomplete by definition).

        Everything else is reported. In particular, PDFs that duplicate a
        registered document and vectors that match no indexed document are
        left in place (orphan vectors are inert, see ``retrieval_filters``)
        unless ``PRUNE_ORPHAN_VECTORS=true``.
        """

        report: dict = {
            "completed_deletions": [],
            "reindexed": [],
            "failed": [],
            "adopted_files": [],
            "duplicate_files": [],
            "removed_partial_files": [],
            "recovered_replacements": [],
            "pending_cleanups": [],
            "orphan_vectors": {},
            "pruned_orphan_vectors": {},
        }

        with self._lock:
            self._chunks_by_source.clear()
            upload_dir = self.settings.upload_dir

            for part in upload_dir.glob("*.part"):
                part.unlink(missing_ok=True)
                report["removed_partial_files"].append(part.name)

            for record in self.registry.list_by_status("deleting"):
                try:
                    self._remove_everywhere(record)
                    report["completed_deletions"].append(record["original_filename"])
                except Exception:
                    logger.exception("Could not finish deleting %s", record["document_id"])

            self._recover_operations(report)

            for record in self.registry.list_documents():
                if record["status"] == "deleting":
                    # Deletion could not be finished above (e.g. file locked).
                    # Never re-index it; it stays 'deleting' until it can be removed.
                    report["failed"].append(
                        {
                            "document": record["original_filename"],
                            "reason": "deletion still pending",
                        }
                    )
                    continue

                self._reconcile_record(record, report)

            self._adopt_untracked_files(report)
            self._handle_orphan_vectors(report)
            self._rebuild_bm25()

            self.last_reconcile_report = report

        logger.info(
            "Reconcile finished: %d indexed, %d failed, %d duplicate files, %d orphan sources",
            len(self._chunks_by_source),
            len(report["failed"]),
            len(report["duplicate_files"]),
            len(report["orphan_vectors"]),
        )

        return report

    def _recover_operations(self, report: dict) -> None:
        """Finish or undo journalled replacements. Safe to repeat."""

        for operation in self.registry.list_operations():
            label = f"{operation['old_stored_filename']} -> {operation['new_stored_filename']}"

            try:
                if operation["state"] == "cleanup_pending":
                    # Registry already points at the new version.
                    self._finish_replace_cleanup(operation)
                else:
                    # Crashed before the switch: old version is still active,
                    # so the half-built new version is discarded.
                    new_stored = operation["new_stored_filename"]

                    if self.registry.get_by_source(new_stored) is None:
                        self._discard_new_version(new_stored)
                        self._chunks_by_source.pop(new_stored, None)

                    self.registry.finish_operation(operation["operation_id"])

                report["recovered_replacements"].append(label)

            except Exception:
                logger.exception("Could not recover replacement %s", label)
                report["pending_cleanups"].append(label)

    def _pending_operation_files(self) -> set[str]:
        """Filenames owned by unfinished journal entries (never adopt these)."""

        names: set[str] = set()

        for operation in self.registry.list_operations():
            names.add(operation["old_stored_filename"])
            names.add(operation["new_stored_filename"])

        return names

    def _reconcile_record(self, record: dict, report: dict) -> None:
        path = self.settings.upload_dir / record["stored_filename"]
        document_id = record["document_id"]

        if not path.exists():
            self.registry.update_status(
                document_id,
                status="failed",
                error_message="PDF file is missing from storage.",
            )
            report["failed"].append(
                {"document": record["original_filename"], "reason": "file missing"}
            )
            return

        try:
            pages, chunks = self._extract_and_chunk(path)
            add_documents(self.vector_store, chunks)  # add-only-missing
        except Exception as error:
            logger.exception("Could not index %s", record["original_filename"])
            self.registry.update_status(
                document_id, status="failed", error_message=str(error)
            )
            report["failed"].append(
                {"document": record["original_filename"], "reason": str(error)}
            )
            return

        self._chunks_by_source[record["stored_filename"]] = chunks

        if (
            record["status"] != "indexed"
            or record["page_count"] != pages
            or record["chunk_count"] != len(chunks)
        ):
            self.registry.update_status(
                document_id,
                status="indexed",
                page_count=pages,
                chunk_count=len(chunks),
            )
            report["reindexed"].append(record["original_filename"])

    def _adopt_untracked_files(self, report: dict) -> None:
        """Register PDFs that exist on disk but not in the registry."""

        known = {item["stored_filename"] for item in self.registry.list_documents()}
        known |= self._pending_operation_files()

        for path in sorted(self.settings.upload_dir.glob("*.pdf")):
            if path.name in known:
                continue

            file_hash = sha256_file(path)
            duplicate = self.registry.get_by_hash(file_hash)

            if duplicate is not None:
                logger.warning(
                    "%s duplicates %s; leaving it on disk untouched",
                    path.name,
                    duplicate["stored_filename"],
                )
                report["duplicate_files"].append(
                    {
                        "file": path.name,
                        "duplicate_of": duplicate["stored_filename"],
                        "document_id": duplicate["document_id"],
                    }
                )
                continue

            try:
                record = self.registry.create_document(
                    document_id=str(uuid4()),
                    original_filename=_legacy_display_name(path.name),
                    stored_filename=path.name,
                    file_hash=file_hash,
                    file_size=path.stat().st_size,
                )
            except (DuplicateDocumentError, ValueError):
                logger.exception("Could not register %s", path.name)
                continue

            self._reconcile_record(record, report)
            report["adopted_files"].append(path.name)

    def _handle_orphan_vectors(self, report: dict) -> None:
        indexed = {
            item["stored_filename"]
            for item in self.registry.list_by_status("indexed")
        }

        orphans = {
            source: count
            for source, count in count_chunks_by_source(self.vector_store).items()
            if source not in indexed
        }

        report["orphan_vectors"] = orphans

        if not orphans:
            return

        if not self.settings.prune_orphan_vectors:
            logger.warning(
                "Vectors without an indexed document were found (kept, not searched): %s. "
                "Set PRUNE_ORPHAN_VECTORS=true to delete them.",
                orphans,
            )
            return

        for source in orphans:
            report["pruned_orphan_vectors"][source] = delete_source(
                self.vector_store, source
            )

        logger.warning("Pruned orphan vectors: %s", report["pruned_orphan_vectors"])

    # ---------------- explicit cleanup (administrative) ----------------

    def plan_cleanup(self) -> dict:
        """
        Read-only: list what ``apply_cleanup`` WOULD remove, with the evidence.

        * duplicate files: a PDF on disk that no registry record uses, whose
          SHA-256 equals a registered document whose own file still exists;
        * orphan vectors: vectors whose ``source`` is used by no registry
          record and no unfinished journal entry. Each source is re-queried
          from ChromaDB and every returned vector's metadata must carry
          exactly that source.
        """

        with self._lock:
            records = self.registry.list_documents()
            registered = {r["stored_filename"] for r in records}
            protected = registered | self._pending_operation_files()
            by_hash = {r["file_hash"]: r for r in records}

            duplicate_files = []
            duplicate_names = set()

            for path in sorted(self.settings.upload_dir.glob("*.pdf")):
                if path.name in protected:
                    continue

                original = by_hash.get(sha256_file(path))

                if original is None:
                    continue  # unknown content: never a cleanup candidate

                if not (self.settings.upload_dir / original["stored_filename"]).exists():
                    continue  # the registered copy is gone; keep this one

                duplicate_names.add(path.name)
                duplicate_files.append(
                    {
                        "file": path.name,
                        "size": path.stat().st_size,
                        "duplicate_of": {
                            "document_id": original["document_id"],
                            "original_filename": original["original_filename"],
                            "stored_filename": original["stored_filename"],
                        },
                    }
                )

            orphan_vectors = []

            for source, count in sorted(count_chunks_by_source(self.vector_store).items()):
                if source in protected:
                    continue

                found = self.vector_store._collection.get(
                    where={"source": source}, include=["metadatas"]
                )
                verified = len(found["ids"]) == count and all(
                    (meta or {}).get("source") == source
                    for meta in found["metadatas"]
                )

                orphan_vectors.append(
                    {
                        "source": source,
                        "vector_count": len(found["ids"]),
                        "metadata_verified": verified,
                        "duplicate_file_on_disk": source in duplicate_names,
                    }
                )

        return {"duplicate_files": duplicate_files, "orphan_vectors": orphan_vectors}

    def apply_cleanup(self, quarantine_root: Path | None = None) -> dict:
        """
        Apply ``plan_cleanup`` with a recovery path.

        Nothing is deleted outright: duplicate files are MOVED into
        ``<data_dir>/quarantine/<timestamp>/files`` and the text + metadata of
        every removed vector is written to ``.../vectors/*.json`` (re-index by
        re-uploading the PDF). A ``manifest.json`` records everything.
        """

        with self._lock:
            plan = self.plan_cleanup()  # re-verified at apply time

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            root = (quarantine_root or self.settings.data_dir / "quarantine") / stamp
            manifest: dict = {"created_at": stamp, "files": [], "vectors": []}

            for item in plan["duplicate_files"]:
                source_path = self.settings.upload_dir / item["file"]
                target = root / "files" / item["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(target))
                manifest["files"].append({**item, "moved_to": str(target)})

            for item in plan["orphan_vectors"]:
                if not item["metadata_verified"]:
                    logger.warning("Skipping %s: metadata not verified", item["source"])
                    continue

                found = self.vector_store._collection.get(
                    where={"source": item["source"]},
                    include=["documents", "metadatas"],
                )
                backup = root / "vectors" / f"{_UNSAFE_FILENAME_CHARS.sub('_', item['source'])}.json"
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_text(
                    json.dumps(
                        {
                            "source": item["source"],
                            "ids": found["ids"],
                            "documents": found["documents"],
                            "metadatas": found["metadatas"],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                removed = delete_source(self.vector_store, item["source"])
                manifest["vectors"].append(
                    {"source": item["source"], "removed": removed, "backup": str(backup)}
                )

            if manifest["files"] or manifest["vectors"]:
                (root / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )

        return manifest

    # ---------------- BM25 ----------------

    def _rebuild_bm25(self) -> None:
        all_chunks = [
            chunk
            for chunks in self._chunks_by_source.values()
            for chunk in chunks
        ]

        self._set_bm25_documents(all_chunks)
