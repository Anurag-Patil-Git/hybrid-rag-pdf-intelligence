
import sqlite3

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "documents.db"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class DuplicateDocumentError(Exception):
    """Raised when a document with the same hash already exists."""

    def __init__(self, existing_document: dict):
        self.existing_document = existing_document
        super().__init__("This document has already been uploaded.")


class DocumentRegistry:
    """
    Persistent SQLite registry for uploaded PDF documents.

    Responsibilities:
    - Register uploaded documents
    - Detect duplicate files using SHA-256
    - Track document processing status
    - Store document metadata
    - Retrieve document records
    - Delete document records
    """

    VALID_STATUSES = {
        "processing",
        "indexed",
        "failed",
        "deleting",
    }

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):

        self.db_path = Path(db_path)
        self._lock = RLock()

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        """Yield a connection that commits/rolls back and is always closed.

        Closing matters on Windows, where open handles block file deletion.
        """

        connection = sqlite3.connect(
            self.db_path,
            timeout=30
        )

        connection.row_factory = sqlite3.Row

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self):
        """Create the documents table and indexes."""

        with self._lock:

            with self._get_connection() as conn:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,

                        original_filename TEXT NOT NULL,
                        stored_filename TEXT NOT NULL UNIQUE,

                        file_hash TEXT NOT NULL UNIQUE,
                        file_size INTEGER NOT NULL,

                        page_count INTEGER NOT NULL DEFAULT 0,
                        chunk_count INTEGER NOT NULL DEFAULT 0,

                        status TEXT NOT NULL DEFAULT 'processing',

                        error_message TEXT,

                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Journal of multi-step operations (currently: replace).
                # state 'indexing'        -> new version being built; registry
                #                            still points at the old one.
                # state 'cleanup_pending' -> registry points at the new version;
                #                            old file/vectors still to remove.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS operations (
                        operation_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        state TEXT NOT NULL,
                        old_stored_filename TEXT,
                        new_stored_filename TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_documents_status
                    ON documents(status)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_documents_filename
                    ON documents(original_filename)
                """)

    @staticmethod
    def _row_to_dict(row):
        return dict(row) if row else None

    def create_document(
        self,
        document_id: str,
        original_filename: str,
        stored_filename: str,
        file_hash: str,
        file_size: int
    ) -> dict:
        """
        Register a new document before processing.

        SHA-256 and stored filename must be unique.
        """

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not original_filename.strip():
            raise ValueError("Original filename cannot be empty.")

        if not stored_filename.strip():
            raise ValueError("Stored filename cannot be empty.")

        if not file_hash.strip():
            raise ValueError("File hash cannot be empty.")

        if file_size <= 0:
            raise ValueError("File size must be positive.")

        timestamp = current_timestamp()

        with self._lock:

            with self._get_connection() as conn:

                existing = conn.execute("""
                    SELECT *
                    FROM documents
                    WHERE file_hash = ?
                """, (file_hash,)).fetchone()

                if existing:
                    raise DuplicateDocumentError(
                        self._row_to_dict(existing)
                    )

                conn.execute("""
                    INSERT INTO documents (
                        document_id,
                        original_filename,
                        stored_filename,
                        file_hash,
                        file_size,
                        page_count,
                        chunk_count,
                        status,
                        error_message,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 0, 0, 'processing', NULL, ?, ?)
                """, (
                    document_id,
                    original_filename,
                    stored_filename,
                    file_hash,
                    file_size,
                    timestamp,
                    timestamp
                ))

                row = conn.execute("""
                    SELECT *
                    FROM documents
                    WHERE document_id = ?
                """, (document_id,)).fetchone()

                return self._row_to_dict(row)

    def get_by_id(self, document_id: str):
        """Retrieve a document using its ID."""

        with self._lock:

            with self._get_connection() as conn:

                row = conn.execute("""
                    SELECT *
                    FROM documents
                    WHERE document_id = ?
                """, (document_id,)).fetchone()

        return self._row_to_dict(row)

    def get_by_hash(self, file_hash: str):
        """Find a document using its SHA-256 hash."""

        with self._lock:

            with self._get_connection() as conn:

                row = conn.execute("""
                    SELECT *
                    FROM documents
                    WHERE file_hash = ?
                """, (file_hash,)).fetchone()

        return self._row_to_dict(row)

    def get_by_source(self, stored_filename: str):
        """Retrieve a document using its stored filename."""

        with self._lock:

            with self._get_connection() as conn:

                row = conn.execute("""
                    SELECT *
                    FROM documents
                    WHERE stored_filename = ?
                """, (stored_filename,)).fetchone()

        return self._row_to_dict(row)

    def update_status(
        self,
        document_id: str,
        status: str,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None
    ):
        """Update processing status and optional metadata."""

        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid document status: {status}"
            )

        if page_count is not None and page_count < 0:
            raise ValueError("Page count cannot be negative.")

        if chunk_count is not None and chunk_count < 0:
            raise ValueError("Chunk count cannot be negative.")

        timestamp = current_timestamp()

        with self._lock:

            with self._get_connection() as conn:

                result = conn.execute("""
                    UPDATE documents
                    SET
                        status = ?,
                        page_count = COALESCE(?, page_count),
                        chunk_count = COALESCE(?, chunk_count),
                        error_message = ?,
                        updated_at = ?
                    WHERE document_id = ?
                """, (
                    status,
                    page_count,
                    chunk_count,
                    error_message,
                    timestamp,
                    document_id
                ))

                return result.rowcount > 0

    def list_documents(self):
        """Return all registered documents, newest first."""

        with self._lock:

            with self._get_connection() as conn:

                rows = conn.execute("""
                    SELECT *
                    FROM documents
                    ORDER BY created_at DESC
                """).fetchall()

        return [
            self._row_to_dict(row)
            for row in rows
        ]

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document registry record.

        Physical PDF and ChromaDB deletion will be handled
        by the document lifecycle service/API layer.
        """

        with self._lock:

            with self._get_connection() as conn:

                result = conn.execute("""
                    DELETE FROM documents
                    WHERE document_id = ?
                """, (document_id,))

                return result.rowcount > 0

    def list_by_status(self, status: str):
        """Return all documents currently in the given status."""

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid document status: {status}")

        with self._lock:

            with self._get_connection() as conn:

                rows = conn.execute("""
                    SELECT *
                    FROM documents
                    WHERE status = ?
                    ORDER BY created_at DESC
                """, (status,)).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def replace_file(
        self,
        document_id: str,
        original_filename: str,
        stored_filename: str,
        file_hash: str,
        file_size: int,
        page_count: int,
        chunk_count: int,
        operation_id: str | None = None
    ) -> dict | None:
        """
        Atomically point an existing record at a new file.

        If operation_id is given, the journal entry moves to
        'cleanup_pending' in the same transaction, so the switch and the
        "old version needs cleanup" fact can never be out of step.

        The record keeps its document_id and created_at; status becomes
        'indexed'. Raises sqlite3.IntegrityError if the hash or stored
        filename already belongs to another record.
        """

        timestamp = current_timestamp()

        with self._lock:

            with self._get_connection() as conn:

                result = conn.execute("""
                    UPDATE documents
                    SET
                        original_filename = ?,
                        stored_filename = ?,
                        file_hash = ?,
                        file_size = ?,
                        page_count = ?,
                        chunk_count = ?,
                        status = 'indexed',
                        error_message = NULL,
                        updated_at = ?
                    WHERE document_id = ?
                """, (
                    original_filename,
                    stored_filename,
                    file_hash,
                    file_size,
                    page_count,
                    chunk_count,
                    timestamp,
                    document_id
                ))

                if result.rowcount == 0:
                    return None

                if operation_id is not None:
                    conn.execute("""
                        UPDATE operations
                        SET state = 'cleanup_pending', updated_at = ?
                        WHERE operation_id = ?
                    """, (timestamp, operation_id))

                row = conn.execute("""
                    SELECT * FROM documents WHERE document_id = ?
                """, (document_id,)).fetchone()

        return self._row_to_dict(row)

    # ---------------- operation journal ----------------

    def begin_replace_operation(
        self,
        operation_id: str,
        document_id: str,
        old_stored_filename: str,
        new_stored_filename: str
    ) -> None:
        """Record that a replacement is starting (state 'indexing')."""

        timestamp = current_timestamp()

        with self._lock:

            with self._get_connection() as conn:

                conn.execute("""
                    INSERT INTO operations (
                        operation_id, document_id, kind, state,
                        old_stored_filename, new_stored_filename,
                        created_at, updated_at
                    )
                    VALUES (?, ?, 'replace', 'indexing', ?, ?, ?, ?)
                """, (
                    operation_id,
                    document_id,
                    old_stored_filename,
                    new_stored_filename,
                    timestamp,
                    timestamp
                ))

    def list_operations(self) -> list[dict]:
        """Return unfinished journal entries, oldest first."""

        with self._lock:

            with self._get_connection() as conn:

                rows = conn.execute("""
                    SELECT * FROM operations ORDER BY created_at
                """).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def finish_operation(self, operation_id: str) -> bool:
        """Remove a completed (or abandoned) journal entry."""

        with self._lock:

            with self._get_connection() as conn:

                result = conn.execute("""
                    DELETE FROM operations WHERE operation_id = ?
                """, (operation_id,))

                return result.rowcount > 0
