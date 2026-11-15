
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import get_embedding_model
from backend.vector_store import create_vector_store, add_documents
from backend.bm25_retriever import BM25Retriever
from backend.hybrid_retriever import HybridRetriever
from backend.mmr import select_mmr_documents
from backend.reranker import get_reranker, rerank_documents
from backend.context_builder import build_context


# Load and process PDF
documents = extract_pdf_pages("data/uploads/sample.pdf")
chunks = split_documents(documents)

# Initialize retrieval components
embedding_model = get_embedding_model()
vector_store = create_vector_store(embedding_model)
add_documents(vector_store, chunks)

bm25_retriever = BM25Retriever(chunks)

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever
)

query = "What are LangChain indexes?"

# Hybrid retrieval
hybrid_results = hybrid_retriever.search(
    query=query,
    top_k=5,
    fetch_k=15,
    filters={"source": "sample.pdf"}
)

# MMR
mmr_results = select_mmr_documents(
    query=query,
    results=hybrid_results,
    embedding_model=embedding_model,
    top_k=5
)

# Cross-Encoder reranking
reranker = get_reranker()

reranked_documents = rerank_documents(
    query=query,
    documents=[document for document, _ in mmr_results],
    reranker=reranker,
    top_n=3
)

# Context construction
context, citations = build_context(
    documents=reranked_documents,
    max_context_chars=12000
)

print("\nGenerated Context:\n")
print(context)

print("\nCitation Metadata:\n")
for citation in citations:
    print(citation)

print("\nTotal context characters:", len(context))
