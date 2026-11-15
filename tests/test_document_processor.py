from backend.document_processor import extract_pdf_pages

pdf_path = "data/uploads/sample.pdf"

documents = extract_pdf_pages(pdf_path)

print("Total extracted pages:", len(documents))

for doc in documents[:2]:
    print("\nMetadata:", doc.metadata)
    print("Extracted text:", doc.page_content[:300])