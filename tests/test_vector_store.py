
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import get_embedding_model
from backend.vector_store import (
    create_vector_store,
    add_documents,
    search_documents
)


# STEP 1: Extract PDF
print("Extracting PDF...")

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


# STEP 3: Load embedding model
print("\nLoading embedding model...")

embedding_model = get_embedding_model()

print("Embedding model loaded!")


# STEP 4: Initialize ChromaDB
print("\nInitializing ChromaDB...")

vector_store = create_vector_store(
    embedding_model
)

print("Vector database initialized!")


# STEP 5: Store chunks
print("\nStoring document chunks...")

add_documents(
    vector_store,
    chunks
)

print("Documents stored successfully!")


# STEP 6: Semantic search
query = "What is the main topic discussed in this document?"

print("\nSearching for:", query)

results = search_documents(
    vector_store,
    query,
    k=3
)


# STEP 7: Display results
print("\nRetrieved Results")
print("=" * 50)

for index, (document, score) in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("Source:", document.metadata["source"])
    print("Page:", document.metadata["page"])
    print("Chunk ID:", document.metadata["chunk_id"])
    print("Distance:", round(score, 4))
    print("Content:", document.page_content[:300])


# STEP 8: Check collection count
collection = vector_store._collection

print("\nTotal stored vectors:", collection.count())
