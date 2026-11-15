
import logging

from backend.embedding import get_embedding_model
from backend.vector_store import create_vector_store, add_documents
from backend.bm25_retriever import BM25Retriever
from backend.hybrid_retriever import HybridRetriever
from backend.mmr import select_mmr_documents
from backend.reranker import get_reranker, rerank_documents
from backend.context_builder import apply_citation_labels, build_context
from backend.llm import get_llm, generate_answer, NO_ANSWER
from backend.query_rewriter import rewrite_query
from backend.conversation import ConversationManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete conversational RAG pipeline.

    Components:
    - ChromaDB
    - BM25
    - Reciprocal Rank Fusion
    - MMR
    - Cross-Encoder reranking
    - Context construction
    - Qwen LLM
    - Conversation management
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        bm25_retriever,
        reranker,
        llm,
        conversation_manager
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.llm = llm
        self.conversation_manager = conversation_manager

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            rrf_k=60
        )

    @classmethod
    def create(
        cls,
        conversation_manager=None,
        vector_db_dir=None
    ):
        """
        Load all models and connect to ChromaDB without indexing anything.

        The BM25 index starts empty (None) and is supplied later with
        set_bm25_documents().
        """

        logger.info("Loading embedding model, ChromaDB, reranker and LLM...")

        embedding_model = get_embedding_model()
        vector_store = create_vector_store(
            embedding_model,
            persist_directory=vector_db_dir
        )

        if conversation_manager is None:
            conversation_manager = ConversationManager(max_messages=20)

        return cls(
            embedding_model=embedding_model,
            vector_store=vector_store,
            bm25_retriever=None,
            reranker=get_reranker(),
            llm=get_llm(),
            conversation_manager=conversation_manager
        )

    def set_bm25_documents(self, chunks) -> None:
        """Rebuild the in-memory BM25 index (None when there are no chunks)."""

        retriever = BM25Retriever(documents=chunks) if chunks else None

        self.bm25_retriever = retriever
        self.hybrid_retriever.bm25_retriever = retriever

    @classmethod
    def from_documents(
        cls,
        chunks,
        conversation_manager=None
    ):
        """
        Initialize all models and indexes from document chunks.

        Call this once when initializing the application or
        rebuilding the retrieval index.
        """

        if not chunks:
            raise ValueError("Cannot initialize RAG without document chunks.")

        print("Initializing embedding model...")
        embedding_model = get_embedding_model()

        print("Initializing ChromaDB...")
        vector_store = create_vector_store(embedding_model)

        print("Storing document chunks...")
        add_documents(vector_store, chunks)

        print("Initializing BM25...")
        bm25_retriever = BM25Retriever(documents=chunks)

        print("Loading Cross-Encoder...")
        reranker = get_reranker()

        print("Loading Qwen LLM...")
        llm = get_llm()

        if conversation_manager is None:
            conversation_manager = ConversationManager(
                max_messages=20
            )

        print("RAG pipeline initialized successfully.")

        return cls(
            embedding_model=embedding_model,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            llm=llm,
            conversation_manager=conversation_manager
        )

    def ask(
        self,
        question: str,
        session_id: str,
        filters: dict | None = None,
        citation_labels: dict | None = None,
        retrieval_k: int = 5,
        fetch_k: int = 15,
        final_k: int = 3
    ):
        """
        Execute the complete conversational RAG workflow.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty.")

        if retrieval_k <= 0 or fetch_k <= 0 or final_k <= 0:
            raise ValueError("Retrieval limits must be greater than zero.")

        if retrieval_k > fetch_k:
            raise ValueError("retrieval_k cannot exceed fetch_k.")

        # STEP 1: Retrieve conversation history
        chat_history = self.conversation_manager.get_history(
            session_id
        )

        # STEP 2: Rewrite follow-up questions
        standalone_question = rewrite_query(
            question=question,
            chat_history=chat_history,
            llm=self.llm
        )

        # STEP 3: Hybrid retrieval
        hybrid_results = self.hybrid_retriever.search(
            query=standalone_question,
            top_k=retrieval_k,
            fetch_k=fetch_k,
            filters=filters
        )

        # Dense search and BM25 already apply ``filters``. Drop anything
        # outside the selected documents again so MMR, reranking and the
        # context can only see that set.
        if citation_labels is not None:
            hybrid_results = [
                (document, score)
                for document, score in hybrid_results
                if document.metadata.get("source") in citation_labels
            ]

        # STEP 4: MMR selection
        mmr_results = select_mmr_documents(
            query=standalone_question,
            results=hybrid_results,
            embedding_model=self.embedding_model,
            top_k=min(retrieval_k, len(hybrid_results)),
            lambda_mult=0.7
        )

        # STEP 5: Cross-Encoder reranking
        reranked_documents = rerank_documents(
            query=standalone_question,
            documents=[
                document for document, _ in mmr_results
            ],
            reranker=self.reranker,
            top_n=final_k
        )

        # STEP 6: Build context and citations
        context, citations = build_context(
            documents=reranked_documents,
            max_context_chars=12000
        )

        # STEP 7: Generate grounded answer
        answer = generate_answer(
            question=question,
            context=context,
            llm=self.llm
        )

        # A no-answer response has nothing to cite. The HTTP body and the
        # stored message must agree, or a reload would hide sources the
        # live reply just showed.
        visible_citations = [] if answer.strip() == NO_ANSWER else citations
        visible_citations = apply_citation_labels(visible_citations, citation_labels)

        # STEP 8: Update conversation history
        self.conversation_manager.create_session(session_id)

        self.conversation_manager.add_message(
            session_id=session_id,
            role="user",
            content=question
        )

        self.conversation_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata={
                "citations": visible_citations,
                "rewritten_question": standalone_question,
                "retrieved_chunks": len(hybrid_results),
                "selected_chunks": len(reranked_documents),
                "context_characters": len(context)
            }
        )

        # STEP 9: Return structured response
        return {
            "session_id": session_id,
            "question": question,
            "rewritten_question": standalone_question,
            "answer": answer,
            "citations": visible_citations,
            "retrieved_chunks": len(hybrid_results),
            "selected_chunks": len(reranked_documents),
            "context_characters": len(context)
        }
