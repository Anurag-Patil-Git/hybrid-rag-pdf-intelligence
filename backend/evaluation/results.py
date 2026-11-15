"""Persist and load evaluation run JSON without overwriting prior runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "evaluation" / "results"
)


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid4().hex[:8]}"


def results_dir(path: str | Path | None = None) -> Path:
    directory = Path(path) if path else DEFAULT_RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_run(payload: dict, directory: str | Path | None = None) -> Path:
    """Write one run to a new file. Never overwrites an existing path."""

    run_id = payload.get("run_id") or new_run_id()
    payload = {**payload, "run_id": run_id}
    target = results_dir(directory) / f"{run_id}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation run: {target}")
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs(directory: str | Path | None = None) -> list[dict]:
    """Return run summaries newest-first."""

    folder = results_dir(directory)
    runs = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        try:
            payload = load_run(path)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append({
            "path": str(path),
            "run_id": payload.get("run_id"),
            "created_at": payload.get("created_at"),
            "dataset_version": payload.get("dataset_version"),
            "kind": payload.get("kind"),
            "top_k": payload.get("top_k"),
            "modes": list((payload.get("aggregates") or {}).keys()),
            "verified_questions": payload.get("verified_questions"),
        })
    return runs


def latest_run(directory: str | Path | None = None) -> dict | None:
    runs = list_runs(directory)
    if not runs:
        return None
    return load_run(runs[0]["path"])


def load_manual_reviews(path: str | Path | None) -> dict[str, dict]:
    """Optional reviewer file keyed by question_id."""

    if path is None:
        return {}
    review_path = Path(path)
    if not review_path.exists():
        return {}
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    reviews = payload.get("reviews") if isinstance(payload, dict) else payload
    if not isinstance(reviews, list):
        return {}
    return {
        item["question_id"]: item
        for item in reviews
        if isinstance(item, dict) and item.get("question_id")
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
