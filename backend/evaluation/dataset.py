"""Load and validate manually labelled evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "question",
    "verified",
    "expected_answer_criteria",
    "relevant",
}


class DatasetError(ValueError):
    """The evaluation dataset file is invalid."""


def load_dataset(path: str | Path) -> dict:
    """Load a dataset JSON file and validate its structure."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"Dataset not found: {dataset_path}")

    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetError(f"Dataset is not valid JSON: {error}") from error

    return validate_dataset(payload)


def validate_dataset(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise DatasetError("Dataset root must be an object.")

    version = payload.get("dataset_version")
    if not isinstance(version, str) or not version.strip():
        raise DatasetError("dataset_version must be a non-empty string.")

    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise DatasetError("documents must be a non-empty list.")

    doc_keys = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise DatasetError(f"documents[{index}] must be an object.")
        key = document.get("doc_key")
        if not isinstance(key, str) or not key.strip():
            raise DatasetError(f"documents[{index}].doc_key is required.")
        if key in doc_keys:
            raise DatasetError(f"Duplicate doc_key: {key}")
        doc_keys.add(key)
        if not document.get("match_original_filename") and not document.get(
            "match_stored_filename_contains"
        ):
            raise DatasetError(
                f"documents[{index}] needs match_original_filename or "
                "match_stored_filename_contains."
            )

    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise DatasetError("questions must be a list.")

    seen_ids = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise DatasetError(f"questions[{index}] must be an object.")
        missing = REQUIRED_QUESTION_FIELDS - set(question)
        if missing:
            raise DatasetError(
                f"questions[{index}] missing fields: {sorted(missing)}"
            )
        question_id = question["question_id"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise DatasetError(f"questions[{index}].question_id is invalid.")
        if question_id in seen_ids:
            raise DatasetError(f"Duplicate question_id: {question_id}")
        seen_ids.add(question_id)

        if not isinstance(question["question"], str) or not question["question"].strip():
            raise DatasetError(f"questions[{index}].question must be non-empty.")
        if not isinstance(question["verified"], bool):
            raise DatasetError(f"questions[{index}].verified must be a boolean.")
        if not isinstance(question["expected_answer_criteria"], str):
            raise DatasetError(
                f"questions[{index}].expected_answer_criteria must be a string."
            )

        relevant = question["relevant"]
        if not isinstance(relevant, list):
            raise DatasetError(f"questions[{index}].relevant must be a list.")
        if question["verified"] and not relevant:
            raise DatasetError(
                f"questions[{index}] is verified but has no relevant labels."
            )
        for item_index, item in enumerate(relevant):
            _validate_relevant_item(item, index, item_index, doc_keys)

    return payload


def verified_questions(dataset: dict) -> list[dict]:
    return [item for item in dataset["questions"] if item.get("verified") is True]


def dataset_summary(dataset: dict) -> dict:
    questions = dataset.get("questions") or []
    verified = verified_questions(dataset)
    return {
        "dataset_version": dataset.get("dataset_version"),
        "total_questions": len(questions),
        "verified_questions": len(verified),
        "unverified_questions": len(questions) - len(verified),
        "documents": len(dataset.get("documents") or []),
    }


def resolve_doc_key(
    document_records: list[dict],
    dataset_documents: list[dict],
) -> dict[str, dict]:
    """Map ``doc_key`` → registry record using filename matchers.

    Matching is by ``original_filename`` equality/substring or by a substring
    of ``stored_filename``. Document IDs are never hard-coded in the dataset.
    """

    resolved = {}
    for spec in dataset_documents:
        key = spec["doc_key"]
        matches = []
        for record in document_records:
            original = record.get("original_filename") or ""
            stored = record.get("stored_filename") or ""
            if spec.get("match_original_filename"):
                needle = spec["match_original_filename"]
                if original == needle or needle in original:
                    matches.append(record)
                    continue
            if spec.get("match_stored_filename_contains"):
                needle = spec["match_stored_filename_contains"]
                if needle in stored:
                    matches.append(record)
        if not matches:
            raise DatasetError(
                f"No indexed document matched doc_key={key!r}. "
                "Upload the PDF or adjust the dataset matchers."
            )
        if len(matches) > 1:
            names = [item.get("original_filename") for item in matches]
            raise DatasetError(
                f"doc_key={key!r} matched multiple documents: {names}"
            )
        resolved[key] = matches[0]
    return resolved


def chunk_is_relevant(
    chunk_meta: dict,
    relevant_items: list[dict],
    doc_key_to_source: dict[str, str],
) -> bool:
    """Return whether a retrieved chunk matches any labelled relevant item."""

    source = chunk_meta.get("source")
    page = chunk_meta.get("page")
    chunk_id = chunk_meta.get("chunk_id")

    for item in relevant_items:
        expected_source = doc_key_to_source.get(item["doc_key"])
        if expected_source is None or source != expected_source:
            continue

        labelled_chunks = item.get("chunk_ids")
        if labelled_chunks:
            if chunk_id in labelled_chunks:
                return True
            continue

        if page is not None and int(page) == int(item["page"]):
            return True

    return False


def labelled_relevant_keys(
    relevant_items: list[dict],
    doc_key_to_source: dict[str, str],
) -> set[tuple]:
    """Stable keys used as the recall denominator.

    When ``chunk_ids`` are present, each chunk id is one relevant unit.
    Otherwise each ``(source, page)`` pair is one relevant unit.
    """

    keys = set()
    for item in relevant_items:
        source = doc_key_to_source.get(item["doc_key"])
        if source is None:
            continue
        chunk_ids = item.get("chunk_ids") or []
        if chunk_ids:
            for chunk_id in chunk_ids:
                keys.add(("chunk", source, chunk_id))
        else:
            keys.add(("page", source, int(item["page"])))
    return keys


def retrieved_key(chunk_meta: dict, relevant_items: list[dict]) -> tuple | None:
    """Map a retrieved chunk onto the same key space as labelled units."""

    source = chunk_meta.get("source")
    page = chunk_meta.get("page")
    chunk_id = chunk_meta.get("chunk_id")
    uses_chunk_ids = any(item.get("chunk_ids") for item in relevant_items)
    if uses_chunk_ids:
        return ("chunk", source, chunk_id)
    if page is None:
        return None
    return ("page", source, int(page))


def _validate_relevant_item(item, question_index, item_index, doc_keys) -> None:
    prefix = f"questions[{question_index}].relevant[{item_index}]"
    if not isinstance(item, dict):
        raise DatasetError(f"{prefix} must be an object.")
    doc_key = item.get("doc_key")
    if doc_key not in doc_keys:
        raise DatasetError(f"{prefix}.doc_key is unknown: {doc_key!r}")
    if "page" not in item:
        raise DatasetError(f"{prefix}.page is required.")
    try:
        int(item["page"])
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{prefix}.page must be an integer.") from error
    chunk_ids = item.get("chunk_ids")
    if chunk_ids is not None and not isinstance(chunk_ids, list):
        raise DatasetError(f"{prefix}.chunk_ids must be a list or null.")
