"""Centralised, environment-driven configuration.

All paths are built with pathlib so the project runs unchanged on Linux and
Windows. Every value can be overridden with an environment variable (or a
``.env`` file, if python-dotenv is installed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _ollama_base_url() -> str:
    """Same host the Ollama client uses, with a scheme so the health probe can call it."""

    raw = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
    value = raw.strip().rstrip("/")
    if not value:
        return "http://127.0.0.1:11434"
    if "://" not in value:
        return f"http://{value}"
    return value


@dataclass(frozen=True)
class Settings:
    """Application settings. Build with :func:`get_settings`."""

    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", PROJECT_ROOT / "data"))
    vector_db_dir: Path = field(default_factory=lambda: _env_path("VECTOR_DB_DIR", PROJECT_ROOT / "vector_db"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 20))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))
    # Orphan vectors (vectors whose source is not in the registry) are only
    # reported by default. Set to true to delete them during startup.
    prune_orphan_vectors: bool = field(default_factory=lambda: _env_bool("PRUNE_ORPHAN_VECTORS", False))
    ollama_base_url: str = field(default_factory=_ollama_base_url)
    # Short on purpose: a health poll must not wait on a hung model server.
    ollama_health_timeout: float = field(default_factory=lambda: _env_float("OLLAMA_HEALTH_TIMEOUT", 1.0))

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def documents_db(self) -> Path:
        return self.data_dir / "documents.db"

    @property
    def conversations_db(self) -> Path:
        return self.data_dir / "conversations.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    """Return settings reflecting the current environment."""
    return Settings()
