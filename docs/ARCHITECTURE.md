# System architecture

Module names and data flow match the current codebase under `backend/` and `frontend/`.

```mermaid
flowchart TB
  subgraph UI["Streamlit frontend"]
    APP["frontend/app.py\nmultipage shell"]
    DASH["views/dashboard.py"]
    CHAT["views/chat.py"]
    LIB["views/library.py"]
    HIST["views/history.py"]
    EVALUI["views/evaluation.py"]
    API_CLIENT["frontend/api_client.py"]
    APP --> DASH & CHAT & LIB & HIST & EVALUI
    DASH & CHAT & LIB & HIST --> API_CLIENT
  end

  subgraph API["FastAPI backend"]
    MAIN["backend/main.py"]
    DOC_SVC["services/document_service.py"]
    REG["document_registry.py\nSQLite documents.db"]
    CONV["conversation.py\nSQLite conversations.db"]
    PIPE["rag_pipeline.py"]
    MAIN --> DOC_SVC & CONV & PIPE
    DOC_SVC --> REG
  end

  subgraph Ingest["Document ingestion"]
    PROC["document_processor.py\nPyMuPDF"]
    CHUNK["chunking.py"]
    EMB["embedding.py\nall-MiniLM-L6-v2 CPU"]
    VS["vector_store.py\nChromaDB"]
    BM25_BUILD["bm25_retriever.py"]
    DOC_SVC --> PROC --> CHUNK --> EMB --> VS
    CHUNK --> BM25_BUILD
  end

  subgraph Retrieve["Retrieval and generation"]
    QR["query_rewriter.py"]
    DENSE["vector_store.search_documents"]
    BM25["bm25_retriever.search"]
    HYB["hybrid_retriever.py\nRRF"]
    MMR["mmr.py"]
    RERANK["reranker.py\nMiniLM cross-encoder CPU"]
    CTX["context_builder.py"]
    LLM["llm.py\nOllama Qwen 2.5 3B"]
    PIPE --> QR --> HYB
    HYB --> DENSE & BM25
    HYB --> MMR --> RERANK --> CTX --> LLM
    PIPE --> CONV
  end

  subgraph Offline["Offline evaluation"]
    EVALPKG["backend/evaluation/*"]
    DATASET["evaluation/datasets/*.json"]
    RESULTS["evaluation/results/*.json\nlocal only"]
    EVALPKG --> DENSE & BM25 & HYB & RERANK
    DATASET --> EVALPKG --> RESULTS
    EVALUI -.->|reads aggregates JSON| RESULTS
  end

  API_CLIENT -->|HTTP| MAIN
  VS --> DENSE
  BM25_BUILD --> BM25
```

## Request paths

| User action | Primary modules |
|---|---|
| Upload / replace / delete PDF | `main.py` → `DocumentService` → registry + files + Chroma + BM25 rebuild |
| Ask a question | `main.py` → `resolve_search` → `RAGPipeline.ask` |
| Manage chats | `main.py` → `ConversationManager` |
| Run evaluation | `python -m backend.evaluation` (does not call `/chat`) |
