
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import get_embedding_model
from backend.vector_store import create_vector_store, add_documents
from backend.bm25_retriever import BM25Retriever
from backend.hybrid_retriever import HybridRetriever
from backend.mmr import select_mmr_documents

# 1. Load and split the PDF
documents = extract_pdf_pages("data/uploads/sample.pdf")
chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

# 2. Initialize embedding model and vector store
embedding_model = get_embedding_model()
vector_store = create_vector_store(embedding_model)
add_documents(vector_store, chunks)

# 3. Initialize retrievers
bm25_retriever = BM25Retriever(documents=chunks)

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    rrf_k=60
)

# 4. Run hybrid retrieval with a metadata filter
query = "What are LangChain indexes?"

hybrid_results = hybrid_retriever.search(
    query=query,
    top_k=5,
    fetch_k=10,
    filters={"source": "sample.pdf"}
)

# 5. Apply MMR to select diverse results
final_results = select_mmr_documents(
    query=query,
    results=hybrid_results,
    embedding_model=embedding_model,
    top_k=5,
    lambda_mult=0.7
)

# 6. Display results
print("\nMMR Retrieval Results")

for rank, (document, score) in enumerate(final_results, start=1):
    print(f"\nRank: {rank}")
    print(f"RRF Score: {score:.4f}")
    print(f"Source: {document.metadata.get('source')}")
    print(f"Page: {document.metadata.get('page')}")
    print(f"Chunk: {document.metadata.get('chunk_id')}")
    print(f"Content: {document.page_content[:300]}")

# 7. Test a page-range filter
page_filtered_results = hybrid_retriever.search(
    query=query,
    top_k=5,
    fetch_k=10,
    filters={
                "$and": [
                    {"source": "sample.pdf"},
                    {"page": {"$gte": 5}},
                    {"page": {"$lte": 10}}
                ]
            
    }
)

print("\nPage-filtered results")

for document, score in page_filtered_results:
    print(
        f"Page: {document.metadata.get('page')} | "
        f"Score: {score:.4f}"
    )
