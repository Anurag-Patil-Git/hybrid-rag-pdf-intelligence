
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    """Load the Cross-Encoder reranking model."""

    return CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
        max_length=512
    )


def rerank_documents(
    query: str,
    documents: list,
    reranker,
    top_n: int = 5
):
    """
    Rerank documents using query-document relevance scores.

    Higher Cross-Encoder scores indicate stronger relevance.
    These scores are not calibrated probabilities.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    if not documents:
        return []

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False
    )

    scored_documents = list(zip(documents, scores))

    scored_documents.sort(
        key=lambda item: float(item[1]),
        reverse=True
    )

    final_results = []

    for document, score in scored_documents[:top_n]:
        document.metadata["rerank_score"] = float(score)
        final_results.append(document)

    return final_results
