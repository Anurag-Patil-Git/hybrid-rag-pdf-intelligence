
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import get_embedding_model
from backend.vector_store import create_vector_store, add_documents
from backend.bm25_retriever import BM25Retriever
from backend.hybrid_retriever import HybridRetriever
from backend.mmr import select_mmr_documents
from backend.reranker import get_reranker, rerank_documents


# 1. Load PDF
documents = extract_pdf_pages("data/uploads/sample.pdf")

chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

# 2. Initialize embeddings and vector store
embedding_model = get_embedding_model()

vector_store = create_vector_store(embedding_model)

add_documents(vector_store, chunks)

# 3. Initialize retrievers
bm25_retriever = BM25Retriever(chunks)

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    rrf_k=60
)

# 4. Hybrid retrieval
query = "What are LangChain indexes?"

hybrid_results = hybrid_retriever.search(
    query=query,
    top_k=5,
    fetch_k=15,
    filters={"source": "sample.pdf"}
)

# 5. MMR selection
mmr_results = select_mmr_documents(
    query=query,
    results=hybrid_results,
    embedding_model=embedding_model,
    top_k=5,
    lambda_mult=0.7
)

# 6. Load Cross-Encoder
reranker = get_reranker()

# 7. Rerank MMR-selected documents
final_results = rerank_documents(
    query=query,
    documents=[document for document, _ in mmr_results],
    reranker=reranker,
    top_n=3
)

# 8. Display final results
print("\nFinal Cross-Encoder Results")

for rank, document in enumerate(final_results, start=1):
    print(f"\nRank: {rank}")
    print(f"Source: {document.metadata.get('source')}")
    print(f"Page: {document.metadata.get('page')}")
    print(f"Chunk: {document.metadata.get('chunk_id')}")
    print(
        f"Rerank Score: "
        f"{document.metadata.get('rerank_score'):.4f}"
    )
    print(f"Content: {document.page_content[:300]}")

