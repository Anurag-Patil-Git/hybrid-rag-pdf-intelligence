
import json
import sqlite3

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DB_PATH = (
    PROJECT_ROOT / "data" / "conversations.db"
)


def current_timestamp():
    return datetime.now(timezone.utc).isoformat()


class ConversationManager:
    """
    Persistent conversation manager using SQLite.

    Features:
    - Create conversations
    - Store messages
    - Store citations and response metadata
    - Retrieve conversation history
    - List conversations
    - Rename conversations
    - Delete conversations
    - Preserve history across application restarts
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        max_messages: int = 20
    ):

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.db_path = Path(db_path)
        self.max_messages = max_messages
        self._lock = RLock()

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        """Yield a connection that commits/rolls back and is always closed."""

        connection = sqlite3.connect(
            self.db_path,
            timeout=30
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self):
        """Create database tables if they do not exist."""

        with self._lock:

            with self._get_connection() as conn:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT 'New Chat',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL,

                        FOREIGN KEY(session_id)
                        REFERENCES conversations(session_id)
                        ON DELETE CASCADE
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_messages_session_id
                    ON messages(session_id, id)
                """)

    def create_session(
        self,
        session_id: str,
        title: str = "New Chat"
    ):
        """Create a conversation session."""

        if not session_id or not session_id.strip():
            raise ValueError(
                "Session ID cannot be empty."
            )

        timestamp = current_timestamp()

        with self._lock:

            with self._get_connection() as conn:

                conn.execute("""
                    INSERT OR IGNORE INTO conversations
                    (session_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    session_id,
                    title,
                    timestamp,
                    timestamp
                ))

        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None
    ):
        """Persist a conversation message."""

        if not session_id or not session_id.strip():
            raise ValueError(
                "Session ID cannot be empty."
            )

        if role not in {"user", "assistant"}:
            raise ValueError(
                "Role must be user or assistant."
            )

        if not content or not content.strip():
            raise ValueError(
                "Message content cannot be empty."
            )

        timestamp = current_timestamp()

        metadata_json = (
            json.dumps(metadata, ensure_ascii=False)
            if metadata is not None
            else None
        )

        with self._lock:

            with self._get_connection() as conn:

                # Create session if it does not exist.
                conn.execute("""
                    INSERT OR IGNORE INTO conversations
                    (session_id, title, created_at, updated_at)
                    VALUES (?, 'New Chat', ?, ?)
                """, (
                    session_id,
                    timestamp,
                    timestamp
                ))

                conn.execute("""
                    INSERT INTO messages
                    (session_id, role, content, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    role,
                    content.strip(),
                    metadata_json,
                    timestamp
                ))

                # Automatically title a new conversation
                # using its first user message.
                if role == "user":

                    conn.execute("""
                        UPDATE conversations
                        SET title = ?
                        WHERE session_id = ?
                        AND title = 'New Chat'
                    """, (
                        content.strip()[:60],
                        session_id
                    ))

                conn.execute("""
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE session_id = ?
                """, (
                    timestamp,
                    session_id
                ))

    def get_history(self, session_id: str) -> list:
        """
        Return recent messages for query rewriting.

        Only the latest max_messages are returned.
        The complete conversation remains stored in SQLite.
        """

        with self._lock:

            with self._get_connection() as conn:

                rows = conn.execute("""
                    SELECT role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (
                    session_id,
                    self.max_messages
                )).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"]
            }
            for row in reversed(rows)
        ]

    def get_session(self, session_id: str):
        """Retrieve a complete conversation."""

        with self._lock:

            with self._get_connection() as conn:

                session = conn.execute("""
                    SELECT *
                    FROM conversations
                    WHERE session_id = ?
                """, (session_id,)).fetchone()

                if session is None:
                    return None

                rows = conn.execute("""
                    SELECT *
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                """, (session_id,)).fetchall()

        messages = []

        for row in rows:

            metadata = (
                json.loads(row["metadata"])
                if row["metadata"]
                else {}
            )

            messages.append({
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": metadata,
                "created_at": row["created_at"]
            })

        return {
            "session_id": session["session_id"],
            "title": session["title"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "messages": messages
        }

    def list_sessions(self):
        """List all conversations, newest activity first."""

        with self._lock:

            with self._get_connection() as conn:

                rows = conn.execute("""
                    SELECT
                        c.session_id,
                        c.title,
                        c.created_at,
                        c.updated_at,
                        COUNT(m.id) AS message_count

                    FROM conversations c

                    LEFT JOIN messages m
                    ON c.session_id = m.session_id

                    GROUP BY c.session_id

                    ORDER BY c.updated_at DESC
                """).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def rename_session(
        self,
        session_id: str,
        title: str
    ):
        """Rename an existing conversation."""

        if not title or not title.strip():
            raise ValueError(
                "Conversation title cannot be empty."
            )

        with self._lock:

            with self._get_connection() as conn:

                cursor = conn.execute("""
                    UPDATE conversations
                    SET title = ?
                    WHERE session_id = ?
                """, (
                    title.strip()[:100],
                    session_id
                ))

                return cursor.rowcount > 0

    def clear_session(self, session_id: str, reset_title: bool = True) -> bool:
        """Delete messages but preserve the conversation.

        ``reset_title`` keeps the previous behavior for direct callers.
        The API leaves the title alone so Clear and Rename stay separate.
        """

        timestamp = current_timestamp()

        with self._lock:

            with self._get_connection() as conn:

                conn.execute("""
                    DELETE FROM messages
                    WHERE session_id = ?
                """, (session_id,))

                if reset_title:
                    cursor = conn.execute("""
                        UPDATE conversations
                        SET title = 'New Chat',
                            updated_at = ?
                        WHERE session_id = ?
                    """, (
                        timestamp,
                        session_id
                    ))
                else:
                    cursor = conn.execute("""
                        UPDATE conversations
                        SET updated_at = ?
                        WHERE session_id = ?
                    """, (
                        timestamp,
                        session_id
                    ))

                return cursor.rowcount > 0

    def delete_session(self, session_id: str):
        """Delete a conversation and all its messages."""

        with self._lock:

            with self._get_connection() as conn:

                cursor = conn.execute("""
                    DELETE FROM conversations
                    WHERE session_id = ?
                """, (session_id,))

                return cursor.rowcount > 0

    def get_session_count(self) -> int:

        with self._lock:

            with self._get_connection() as conn:

                row = conn.execute("""
                    SELECT COUNT(*)
                    FROM conversations
                """).fetchone()

        return row[0]
