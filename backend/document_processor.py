
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def extract_pdf_pages(file_path: str | Path) -> list[Document]:
    """
    Extract text from a PDF page by page.

    Returns:
        List of LangChain Document objects containing
        page text and source metadata.
    """

    pdf_path = Path(file_path)

    # Validate file
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    documents = []

    # Open PDF
    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            # Extract page text
            text = page.get_text("text").strip()

            # Skip empty pages
            if not text:
                continue

            # Create structured document
            document = Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number
                }
            )

            documents.append(document)

    if not documents:
        raise ValueError(
            "No extractable text found. "
            "The PDF may be scanned or empty."
        )

    return documents
