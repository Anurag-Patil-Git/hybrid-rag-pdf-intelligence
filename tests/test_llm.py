
from backend.document_processor import extract_pdf_pages
from backend.chunking import split_documents
from backend.embedding import get_embedding_model
from backend.vector_store import create_vector_store, add_documents
from backend.bm25_retriever import BM25Retriever
from backend.hybrid_retriever import HybridRetriever
from backend.mmr import select_mmr_documents
from backend.reranker import get_reranker, rerank_documents
from backend.context_builder import build_context
from backend.llm import get_llm, generate_answer


# 1. Load PDF
documents = extract_pdf_pages("data/uploads/sample.pdf")

chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

# 2. Initialize retrieval components
embedding_model = get_embedding_model()

vector_store = create_vector_store(embedding_model)
add_documents(vector_store, chunks)

bm25_retriever = BM25Retriever(chunks)

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever
)

# 3. User question
question = "What are LangChain indexes?"

# 4. Hybrid retrieval
hybrid_results = hybrid_retriever.search(
    query=question,
    top_k=5,
    fetch_k=15,
    filters={"source": "sample.pdf"}
)

# 5. MMR
mmr_results = select_mmr_documents(
    query=question,
    results=hybrid_results,
    embedding_model=embedding_model,
    top_k=5
)

# 6. Cross-Encoder reranking
reranker = get_reranker()

reranked_documents = rerank_documents(
    query=question,
    documents=[document for document, _ in mmr_results],
    reranker=reranker,
    top_n=3
)

# 7. Build context
context, citations = build_context(
    documents=reranked_documents,
    max_context_chars=12000
)

# 8. Initialize Qwen
llm = get_llm()

# 9. Generate answer
answer = generate_answer(
    question=question,
    context=context,
    llm=llm
)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nAVAILABLE CITATIONS:")
for citation in citations:
    print(citation)

