
from collections import defaultdict

from langchain_core.documents import Document

from backend.bm25_retriever import BM25Retriever
from backend.vector_store import search_documents


class HybridRetriever:
    """
    Combines dense retrieval and BM25 using RRF.
    """

    def __init__(
        self,
        vector_store,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    @staticmethod
    def _document_id(document: Document) -> str:
        """
        Generate a stable identity for a document chunk.
        """

        return (
            f"{document.metadata.get('source', 'unknown')}_"
            f"{document.metadata.get('chunk_id', 'unknown')}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 10,
        filters: dict | None = None
    ) -> list[tuple[Document, float]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0 or fetch_k <= 0:
            raise ValueError("Retrieval counts must be positive.")

        if top_k > fetch_k:
            raise ValueError("top_k cannot exceed fetch_k.")

        # 1. Dense retrieval
        dense_results = search_documents(
            vector_store=self.vector_store,
            query=query,
            k=fetch_k,
            filters=filters
        )

        # 2. BM25 retrieval
        # bm25_retriever is None while no document is indexed.
        sparse_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=fetch_k,
                filters=filters
            )
            if self.bm25_retriever is not None
            else []
        )

        # 3. Combine rankings using RRF
        fused_scores = defaultdict(float)
        documents = {}

        retrieval_results = [
            dense_results,
            sparse_results
        ]

        for results in retrieval_results:

            for rank, (document, _) in enumerate(
                results,
                start=1
            ):

                doc_id = self._document_id(document)

                fused_scores[doc_id] += (
                    1 / (self.rrf_k + rank)
                )

                documents[doc_id] = document

        # 4. Sort by fused relevance
        ranked_results = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # 5. Return top results
        return [
            (documents[doc_id], score)
            for doc_id, score in ranked_results[:top_k]
        ]
