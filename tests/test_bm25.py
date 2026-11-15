
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.bm25_retriever import BM25Retriever


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

print("Total chunks:", len(chunks))


# STEP 3: Build BM25 index
print("\nBuilding BM25 index...")

bm25_retriever = BM25Retriever(
    documents=chunks
)

print("BM25 index created successfully!")


# STEP 4: Search
query = "What are LangChain indexes?"

print("\nSearch Query:", query)

results = bm25_retriever.search(
    query=query,
    top_k=5
)


# STEP 5: Display results
print("\n========== BM25 RESULTS ==========")

for rank, (document, score) in enumerate(
    results,
    start=1
):

    print(f"\nRank: {rank}")

    print("Source:", document.metadata["source"])

    print("Page:", document.metadata["page"])

    print("Chunk ID:", document.metadata["chunk_id"])

    print("BM25 Score:", round(score, 4))

    print("Content:", document.page_content[:300])


# STEP 6: Test metadata filtering
print("\n========== FILTERED SEARCH ==========")

filtered_results = bm25_retriever.search(
    query=query,
    top_k=5,
    filters={"source": "sample.pdf"}
)

print("Filtered results:", len(filtered_results))

for document, score in filtered_results:
    print(
        "Page:", document.metadata["page"],
        "| Score:", round(score, 4)
    )
