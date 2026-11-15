
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents

# Step 1: Extract PDF pages
pdf_path = "data/uploads/sample.pdf"

documents = extract_pdf_pages(pdf_path)

print("Original pages:", len(documents))

# Step 2: Generate chunks
chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print("Total chunks generated:", len(chunks))

# Step 3: Inspect first 3 chunks
for chunk in chunks[:3]:

    print("\n-----------------------------")
    print("Chunk ID:", chunk.metadata["chunk_id"])
    print("Source:", chunk.metadata["source"])
    print("Page:", chunk.metadata["page"])
    print("Characters:", len(chunk.page_content))
    print("Content:", chunk.page_content[:200])
