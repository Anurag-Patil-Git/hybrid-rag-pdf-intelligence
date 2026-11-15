"""Retrieval and citation metrics for offline evaluation.

Conventions
-----------
* **Precision@K** = (relevant hits in the top-K list) / K.
  The denominator is always K, even when fewer than K chunks are returned.
  That penalizes short result lists the same way standard IR Precision@K does.

* **Recall@K** = (distinct labelled units recovered in the top-K list) /
  (number of labelled relevant units). A labelled unit is either one
  ``chunk_id`` or one ``(source, page)`` pair. Counting distinct labels
  keeps Recall in ``[0, 1]`` even when several retrieved chunks share a
  page. When a question has no relevant labels, Recall@K is ``None``.

* **MRR** = 1 / rank of the first relevant chunk (1-based). When nothing
  relevant is retrieved, MRR is ``0.0``. When a question has no labels,
  MRR is ``None`` and is excluded from aggregate means.

* **Citation page match** is a proxy: Jaccard overlap between cited pages
  and labelled relevant pages for the same document scope. It does **not**
  prove that the answer is factually supported.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Precision@K. ``relevance`` is a list of per-rank relevance flags."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    hits = sum(1 for flag in list(relevance)[:k] if flag)
    return hits / k


def recall_at_k(
    relevance: Sequence[bool],
    total_relevant: int,
    k: int,
) -> float | None:
    """Recall@K. Returns ``None`` when there are no labelled relevant items."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative.")
    if total_relevant == 0:
        return None

    hits = sum(1 for flag in list(relevance)[:k] if flag)
    return hits / total_relevant


def mean_reciprocal_rank(relevance: Sequence[bool]) -> float | None:
    """MRR for one query. ``None`` when relevance labels are absent.

    Callers that have labels but no hits should pass a non-empty list of
    ``False`` values (or any list without a ``True``) and receive ``0.0``.
    Pass an empty relevance list only when the question itself is unlabelled;
    that yields ``None``.
    """

    if not relevance:
        return None

    for rank, flag in enumerate(relevance, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def average(values: Iterable[float | None]) -> float | None:
    """Mean of defined values. ``None`` when every input is undefined."""

    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def citation_page_match(
    cited_pages: Sequence[int | str],
    relevant_pages: Sequence[int | str],
) -> dict:
    """Page-overlap proxy between cited pages and labelled relevant pages.

    Returns counts plus a Jaccard score. Missing labels yield
    ``score: null`` so the dashboard can show “not measured”.
    """

    cited = {_normalize_page(page) for page in cited_pages if page is not None}
    relevant = {_normalize_page(page) for page in relevant_pages if page is not None}
    cited.discard(None)
    relevant.discard(None)

    if not relevant:
        return {
            "cited_pages": sorted(cited),
            "relevant_pages": [],
            "intersection": [],
            "score": None,
            "note": "No labelled relevant pages; citation match not measured.",
        }

    intersection = cited & relevant
    union = cited | relevant
    score = (len(intersection) / len(union)) if union else 0.0

    return {
        "cited_pages": sorted(cited),
        "relevant_pages": sorted(relevant),
        "intersection": sorted(intersection),
        "score": score,
        "note": (
            "Citation page match is a proxy based on page overlap. "
            "It does not prove factual support."
        ),
    }


def _normalize_page(page) -> int | str | None:
    if page is None or page == "":
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return str(page)
