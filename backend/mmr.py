
import numpy as np


def select_mmr_documents(
    query: str,
    results: list,
    embedding_model,
    top_k: int = 5,
    lambda_mult: float = 0.7
):
    """
    Select relevant but diverse documents from hybrid retrieval results.

    results: list of (Document, RRF score) tuples.
    lambda_mult: higher values prioritize relevance;
                 lower values prioritize diversity.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not results:
        return []

    top_k = min(top_k, len(results))

    documents = [document for document, _ in results]
    original_scores = [score for _, score in results]

    query_embedding = np.asarray(
        embedding_model.embed_query(query),
        dtype=np.float32
    )

    document_embeddings = np.asarray(
        embedding_model.embed_documents(
            [document.page_content for document in documents]
        ),
        dtype=np.float32
    )

    # Normalize vectors so dot products represent cosine similarity.
    query_embedding /= max(np.linalg.norm(query_embedding), 1e-12)

    norms = np.linalg.norm(
        document_embeddings,
        axis=1,
        keepdims=True
    )
    document_embeddings /= np.maximum(norms, 1e-12)

    relevance_scores = document_embeddings @ query_embedding
    similarity_matrix = document_embeddings @ document_embeddings.T

    selected_indices = []
    remaining_indices = list(range(len(documents)))

    # Select the most query-relevant document first.
    first_index = int(np.argmax(relevance_scores))
    selected_indices.append(first_index)
    remaining_indices.remove(first_index)

    while remaining_indices and len(selected_indices) < top_k:
        best_index = None
        best_score = float("-inf")

        for candidate_index in remaining_indices:
            max_similarity = max(
                similarity_matrix[candidate_index][selected_index]
                for selected_index in selected_indices
            )

            mmr_score = (
                lambda_mult * relevance_scores[candidate_index]
                - (1 - lambda_mult) * max_similarity
            )

            if mmr_score > best_score:
                best_score = float(mmr_score)
                best_index = candidate_index

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return [
        (documents[index], original_scores[index])
        for index in selected_indices
    ]
