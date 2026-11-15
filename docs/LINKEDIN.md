# LinkedIn and resume drafts

Based on the implemented project as of Phase 7. Do not claim production SLAs.

## Concise project description

Local CPU-only RAG PDF chatbot: FastAPI + Streamlit, hybrid dense/BM25 retrieval with RRF and cross-encoder reranking, multi-document chat by document ID, SQLite-backed lifecycle and conversations, plus an offline evaluation harness with Precision@K / Recall@K / MRR on 35 manually labelled questions.

## Resume bullets

1. Built an end-to-end RAG PDF Q&A system on CPU (FastAPI, Streamlit, ChromaDB, BM25, Ollama Qwen 2.5 3B) with grounded answers and page-level citations.
2. Implemented hybrid retrieval (dense + BM25 + Reciprocal Rank Fusion), MMR diversification and MiniLM cross-encoder reranking without requiring a GPU.
3. Designed a journalled document lifecycle (SHA-256 dedupe, replace/delete recovery, startup reconciliation) keeping SQLite, files and vectors consistent.
4. Delivered multi-document chat that filters every retrieval stage by validated `document_ids` and surfaces original filenames, pages and snippets in citations.
5. Created an offline evaluation framework comparing dense, BM25, hybrid and hybrid+rerank on 35 verified questions (hybrid Recall@5 = 1.00, hybrid+rerank MRR = 0.91) with 104 hermetic automated tests.

## LinkedIn project entry

**Title:** RAG PDF Intelligence — Local Hybrid Retrieval Chatbot  

**Description:**  
Portfolio capstone: ask questions over your own PDFs on a CPU-only laptop. The stack uses FastAPI, Streamlit, ChromaDB, BM25, RRF hybrid fusion, cross-encoder reranking and Ollama (Qwen 2.5 3B). Includes multi-document selection, citation snippets, document lifecycle recovery and an offline evaluation suite with measured Precision@K, Recall@K and MRR.

**Skills:** Python, FastAPI, Streamlit, LangChain, ChromaDB, RAG, Information Retrieval, SQLite, Ollama, pytest

## LinkedIn launch post draft

I just packaged a local RAG PDF chatbot for my portfolio.

It runs entirely on CPU: FastAPI backend, Streamlit UI, ChromaDB + BM25 hybrid retrieval (RRF), MiniLM cross-encoder reranking, and Qwen 2.5 3B via Ollama.

Highlights:
- Multi-document chat filtered by document IDs end-to-end
- Journalled upload/replace/delete with startup reconciliation
- Offline evaluation on 35 manually labelled questions  
  Hybrid Recall@5 = 1.00 · Hybrid+rerank MRR = 0.91
- 104 automated hermetic tests

This is a learning / demo system, not a hosted production product — no auth yet, and answer quality still needs human review for faithfulness.

Happy to share the repo architecture or evaluation notes if useful.
