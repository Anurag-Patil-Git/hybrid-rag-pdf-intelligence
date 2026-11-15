
import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from backend.metadata_filters import matches_metadata

class BM25Retriever:
    """
    Keyword-based document retrieval using BM25Okapi.
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75
    ):
        if not documents:
            raise ValueError("Documents cannot be empty.")

        self.documents = documents

        # Tokenize all document chunks
        self.tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        # Remove empty documents from the searchable corpus
        valid_data = [
            (doc, tokens)
            for doc, tokens in zip(
                self.documents,
                self.tokenized_documents
            )
            if tokens
        ]

        if not valid_data:
            raise ValueError("No searchable text found.")

        self.documents = [item[0] for item in valid_data]
        self.tokenized_documents = [item[1] for item in valid_data]

        # Initialize BM25
        self.bm25 = BM25Okapi(
            self.tokenized_documents,
            k1=k1,
            b=b
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into searchable tokens.
        Preserves technical identifiers containing
        underscores and hyphens.
        """

        return re.findall(
            r"[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*",
            text.lower()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents using BM25 ranking.

        filters:
            Optional metadata equality filters.

        Returns:
            List of (Document, BM25 score).
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 scores for the complete corpus
        scores = self.bm25.get_scores(query_tokens)

        results = []

        for document, score in zip(self.documents, scores):
            if not matches_metadata(document.metadata, filters):
                continue

            results.append((document, float(score)))

        results.sort(key=lambda item: item[1], reverse=True)

        return results[:top_k]