
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import get_embedding_model
from backend.vector_store import (
    create_vector_store,
    add_documents,
    search_documents
)
from backend.reranker import (
    get_reranker,
    rerank_documents
)


# STEP 1: Extract PDF
print("\nExtracting PDF...")

documents = extract_pdf_pages(
    "data/uploads/sample.pdf"
)

print("Pages extracted:", len(documents))


# STEP 2: Generate chunks
print("\nGenerating chunks...")

chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print("Chunks generated:", len(chunks))


# STEP 3: Initialize embedding model
print("\nLoading embedding model...")

embedding_model = get_embedding_model()


# STEP 4: Initialize ChromaDB
vector_store = create_vector_store(
    embedding_model
)


# STEP 5: Store documents
print("\nIndexing documents...")

add_documents(
    vector_store,
    chunks
)

print("Documents indexed successfully!")


# STEP 6: Define a specific query
query = "What are indexes in LangChain?"

print("\nUser Query:", query)


# STEP 7: Initial vector retrieval
print("\nPerforming vector search...")

retrieved_results = search_documents(
    vector_store,
    query,
    k=10
)

retrieved_documents = [
    document
    for document, score in retrieved_results
]

print("Initial retrieved chunks:", len(retrieved_documents))


# Display initial retrieval
print("\n========== BEFORE RERANKING ==========")

for index, (document, distance) in enumerate(
    retrieved_results, start=1
):
    print(f"\nRank: {index}")
    print("Page:", document.metadata["page"])
    print("Vector distance:", round(distance, 4))
    print("Content:", document.page_content[:200])


# STEP 8: Load Cross-Encoder
print("\nLoading Cross-Encoder...")

reranker = get_reranker()

print("Cross-Encoder loaded successfully!")


# STEP 9: Rerank retrieved chunks
print("\nReranking documents...")

reranked_results = rerank_documents(
    query=query,
    documents=retrieved_documents,
    reranker=reranker,
    top_n=3
)


# STEP 10: Display reranked results
print("\n========== AFTER RERANKING ==========")

for index, document in enumerate(
    reranked_results, start=1
):
    print(f"\nRank: {index}")
    print("Page:", document.metadata["page"])
    print(
        "Rerank score:",
        round(document.metadata["rerank_score"], 4)
    )
    print("Content:", document.page_content[:300])


print("\nReranking completed successfully!")
