"""FastAPI application: HTTP routes only. Business logic lives in services."""

from __future__ import annotations

import logging
import urllib.request
from contextlib import asynccontextmanager
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.conversation import ConversationManager
from backend.core.config import Settings, get_settings
from backend.document_registry import DocumentRegistry
from backend.services.document_service import (
    DocumentNotFound,
    DocumentService,
    DocumentUnavailable,
    DuplicateDocument,
    InvalidSearch,
    InvalidUpload,
)
from backend.vector_store import vector_store_reachable

logger = logging.getLogger(__name__)

CHAT_FAILURE_MESSAGE = "Question processing failed."
DEFAULT_OLLAMA_TIMEOUT = 1.0
EMPTY_STATS = {
    "indexed_documents": 0,
    "total_documents": 0,
    "total_pages": 0,
    "total_chunks": 0,
}


def check_ollama(base_url: str, timeout: float = DEFAULT_OLLAMA_TIMEOUT) -> bool:
    """Return whether the Ollama server answers. Never raises.

    The timeout stays short so a hung model server cannot stall a health
    poll. Failure means the model is unavailable, not that the API is down.
    """

    wait = timeout if timeout and timeout > 0 else DEFAULT_OLLAMA_TIMEOUT
    url = f"{base_url.rstrip('/')}/api/version"
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=wait) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except Exception:
        logger.debug("Ollama is not reachable at %s", base_url)
        return False


# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    # Stored filename of one document. Kept for older clients.
    source: str | None = None
    # Document IDs to search. Omit to search every indexed document.
    # An empty list is rejected. Do not send filenames here.
    document_ids: list[str] | None = None


class CreateSessionRequest(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=100)


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


# --------------------------------------------------
# Application factory
# --------------------------------------------------

def _default_pipeline_factory(settings: Settings, conversation_manager: ConversationManager):
    # Imported lazily so importing this module does not load ML libraries.
    from backend.rag_pipeline import RAGPipeline

    return RAGPipeline.create(
        conversation_manager=conversation_manager,
        vector_db_dir=settings.vector_db_dir,
    )


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings, ConversationManager], object] | None = None,
) -> FastAPI:
    """Build the application. Tests pass their own settings and pipeline factory."""

    settings = settings or get_settings()
    pipeline_factory = pipeline_factory or _default_pipeline_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        settings.data_dir.mkdir(parents=True, exist_ok=True)

        conversation_manager = ConversationManager(
            db_path=settings.conversations_db, max_messages=20
        )
        registry = DocumentRegistry(db_path=settings.documents_db)
        pipeline = pipeline_factory(settings, conversation_manager)

        service = DocumentService(
            registry=registry,
            vector_store=pipeline.vector_store,
            settings=settings,
            set_bm25_documents=pipeline.set_bm25_documents,
        )
        service.reconcile()

        app.state.settings = settings
        app.state.conversation_manager = conversation_manager
        app.state.registry = registry
        app.state.rag_pipeline = pipeline
        app.state.document_service = service

        yield

    app = FastAPI(
        title="Advanced RAG PDF Chatbot",
        description="Conversational PDF question-answering API",
        version="1.1.0",
        lifespan=lifespan,
    )

    def service() -> DocumentService:
        return app.state.document_service

    def read_upload(file: UploadFile) -> bytes:
        # Read one byte past the limit so oversized files are rejected
        # without loading arbitrarily large bodies into memory.
        return file.file.read(settings.max_upload_bytes + 1)

    def duplicate_http_error(error: DuplicateDocument) -> HTTPException:
        return HTTPException(
            status_code=409,
            detail={
                "message": str(error),
                "existing_document": error.existing_document,
            },
        )

    # ---------------- health ----------------

    @app.get("/health")
    def health_check():
        """Report the API, Chroma, and Ollama separately.

        ``status`` is ``healthy`` only when all three answer. A model outage
        leaves the API ``degraded`` with ``backend_available`` still true.
        ``unavailable`` means the API could not read its own registry or
        conversation store. The HTTP status stays 200 so a caller can read
        which component failed.
        """

        backend_available = True
        stats = dict(EMPTY_STATS)
        report = {}
        conversations = 0

        try:
            stats = service().stats()
            report = service().last_reconcile_report or {}
        except Exception:
            logger.exception("Document registry health check failed")
            backend_available = False

        try:
            conversations = app.state.conversation_manager.get_session_count()
        except Exception:
            logger.exception("Conversation store health check failed")
            backend_available = False

        pipeline = getattr(app.state, "rag_pipeline", None)
        vector_store = getattr(pipeline, "vector_store", None)
        vector_db_available = (
            vector_store_reachable(vector_store) if vector_store is not None else False
        )

        llm_available = check_ollama(
            settings.ollama_base_url,
            settings.ollama_health_timeout,
        )

        if not backend_available:
            overall = "unavailable"
        elif vector_db_available and llm_available:
            overall = "healthy"
        else:
            overall = "degraded"

        return {
            "status": overall,
            "backend_available": backend_available,
            "vector_db_available": vector_db_available,
            "llm_available": llm_available,
            "pipeline_initialized": pipeline is not None,
            "uploaded_documents": stats["indexed_documents"],
            "total_documents": stats["total_documents"],
            "total_pages": stats["total_pages"],
            "total_chunks": stats["total_chunks"],
            "total_conversations": conversations,
            "orphan_vector_sources": len(report.get("orphan_vectors", {})),
        }

    # ---------------- documents ----------------

    @app.post("/upload", status_code=201)
    def upload_pdf(file: UploadFile = File(...)):
        try:
            document = service().upload(file.filename, read_upload(file))
        except DuplicateDocument as error:
            raise duplicate_http_error(error)
        except InvalidUpload as error:
            raise HTTPException(status_code=error.status_code, detail=str(error))

        return {"message": "PDF uploaded and indexed successfully.", "document": document}

    @app.get("/documents")
    def list_documents():
        documents = service().list_documents()

        return {"total_documents": len(documents), "documents": documents}

    @app.get("/documents/{document_id}")
    def get_document(document_id: str):
        try:
            return service().get_document(document_id)
        except DocumentNotFound as error:
            raise HTTPException(status_code=404, detail=str(error))

    @app.put("/documents/{document_id}")
    def replace_document(document_id: str, file: UploadFile = File(...)):
        try:
            result = service().replace(document_id, file.filename, read_upload(file))
        except DocumentNotFound as error:
            raise HTTPException(status_code=404, detail=str(error))
        except DuplicateDocument as error:
            raise duplicate_http_error(error)
        except InvalidUpload as error:
            raise HTTPException(status_code=error.status_code, detail=str(error))

        message = (
            "Document replaced and re-indexed."
            if result["changed"]
            else "The uploaded file is identical to the current document; nothing changed."
        )

        return {"message": message, **result}

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str):
        try:
            record = service().delete(document_id)
        except DocumentNotFound as error:
            raise HTTPException(status_code=404, detail=str(error))

        return {"message": "Document deleted successfully.", "document": record}

    # ---------------- conversations ----------------

    @app.post("/sessions")
    def create_session(request: CreateSessionRequest):
        session_id = str(uuid4())

        app.state.conversation_manager.create_session(
            session_id=session_id, title=request.title
        )

        return {
            "message": "Conversation created successfully.",
            "session_id": session_id,
            "title": request.title,
        }

    @app.get("/sessions")
    def list_sessions():
        sessions = app.state.conversation_manager.list_sessions()

        return {"total_sessions": len(sessions), "sessions": sessions}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        conversation = app.state.conversation_manager.get_session(session_id)

        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        return conversation

    @app.patch("/sessions/{session_id}")
    def rename_session(session_id: str, request: RenameSessionRequest):
        updated = app.state.conversation_manager.rename_session(
            session_id=session_id, title=request.title
        )

        if not updated:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        return {
            "message": "Conversation renamed successfully.",
            "session_id": session_id,
            "title": request.title,
        }

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        deleted = app.state.conversation_manager.delete_session(session_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        return {"message": "Conversation deleted successfully.", "session_id": session_id}

    @app.delete("/sessions/{session_id}/messages")
    def clear_session_messages(session_id: str):
        cleared = app.state.conversation_manager.clear_session(
            session_id, reset_title=False
        )

        if not cleared:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        return {"message": "Conversation cleared.", "session_id": session_id}

    # ---------------- chat ----------------

    @app.post("/chat")
    def chat(request: ChatRequest):
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Question cannot be empty.")

        if not request.session_id.strip():
            raise HTTPException(status_code=422, detail="Session ID cannot be empty.")

        try:
            filters, citation_labels = service().resolve_search(
                request.source, request.document_ids
            )
        except DocumentUnavailable as error:
            raise HTTPException(status_code=error.status_code, detail=str(error))
        except InvalidSearch as error:
            raise HTTPException(status_code=error.status_code, detail=str(error))

        try:
            return app.state.rag_pipeline.ask(
                question=request.question,
                session_id=request.session_id,
                filters=filters,
                citation_labels=citation_labels,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        except Exception:
            logger.exception("Question processing failed")
            raise HTTPException(status_code=500, detail=CHAT_FAILURE_MESSAGE)

    return app


app = create_app()
