
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.rag_pipeline import RAGPipeline


# STEP 1: Process PDF
print("\nProcessing PDF...")

documents = extract_pdf_pages(
    "data/uploads/sample.pdf"
)

chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Pages extracted: {len(documents)}")
print(f"Chunks generated: {len(chunks)}")


# STEP 2: Initialize complete RAG pipeline
rag = RAGPipeline.from_documents(chunks)


# STEP 3: First question
session_id = "test_session_001"

question_1 = "What are LangChain indexes?"

print("\nQUESTION 1:")
print(question_1)

response_1 = rag.ask(
    question=question_1,
    session_id=session_id,
    filters={"source": "sample.pdf"}
)

print("\nANSWER 1:")
print(response_1["answer"])

print("\nCITATIONS:")
for citation in response_1["citations"]:
    print(citation)


# STEP 4: Follow-up question
question_2 = "What are their advantages?"

print("\nQUESTION 2:")
print(question_2)

response_2 = rag.ask(
    question=question_2,
    session_id=session_id,
    filters={"source": "sample.pdf"}
)

print("\nREWRITTEN QUESTION:")
print(response_2["rewritten_question"])

print("\nANSWER 2:")
print(response_2["answer"])


# STEP 5: Pipeline statistics
print("\nPIPELINE STATISTICS:")

print("Retrieved chunks:", response_2["retrieved_chunks"])
print("Selected chunks:", response_2["selected_chunks"])
print("Context characters:", response_2["context_characters"])

print("\nCOMPLETE RAG PIPELINE TEST FINISHED.")

