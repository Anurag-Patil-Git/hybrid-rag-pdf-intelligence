"""Phase 1: document lifecycle (upload, duplicates, delete, replace, reconcile)."""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.documents import Document

from backend.document_registry import DocumentRegistry
from backend.services import document_service as ds
from backend.services.document_service import (
    DocumentNotFound,
    DocumentService,
    DocumentUnavailable,
    DuplicateDocument,
    InvalidSearch,
    InvalidUpload,
)
from backend.vector_store import count_chunks_by_source
from tests.conftest import GEMINI_PDF, LANGCHAIN_PDF, make_pdf


def vector_counts(service):
    return count_chunks_by_source(service.vector_store)


def pdf_files(settings):
    return sorted(p.name for p in settings.upload_dir.glob("*.pdf"))


# ---------------- upload ----------------

def test_upload_indexes_everywhere(service, settings, pipeline):
    record = service.upload("langchain.pdf", LANGCHAIN_PDF)

    assert record["status"] == "indexed"
    assert record["original_filename"] == "langchain.pdf"
    assert record["page_count"] == 2
    assert record["chunk_count"] > 0
    assert pdf_files(settings) == [record["stored_filename"]]
    assert vector_counts(service) == {record["stored_filename"]: record["chunk_count"]}
    assert len(pipeline.bm25_retriever.documents) == record["chunk_count"]


def test_duplicate_upload_is_rejected_with_existing_document(service, settings):
    first = service.upload("a.pdf", LANGCHAIN_PDF)

    with pytest.raises(DuplicateDocument) as caught:
        service.upload("renamed-copy.pdf", LANGCHAIN_PDF)

    assert caught.value.existing_document["document_id"] == first["document_id"]
    assert len(service.list_documents()) == 1
    assert pdf_files(settings) == [first["stored_filename"]]
    assert sum(vector_counts(service).values()) == first["chunk_count"]


@pytest.mark.parametrize(
    "filename, data, status",
    [
        ("notes.txt", b"%PDF-1.4 fake", 400),
        (None, b"%PDF-1.4 fake", 400),
        ("empty.pdf", b"", 400),
        ("fake.pdf", b"not a pdf at all", 400),
        ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 1), 413),
    ],
)
def test_invalid_uploads_are_rejected(service, filename, data, status):
    with pytest.raises(InvalidUpload) as caught:
        service.upload(filename, data)

    assert caught.value.status_code == status
    assert service.list_documents() == []


@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.4 corrupt body", make_pdf("")],
    ids=["corrupt", "no-text"],
)
def test_unreadable_pdf_leaves_no_trace(service, settings, data):
    with pytest.raises(InvalidUpload) as caught:
        service.upload("bad.pdf", data)

    assert caught.value.status_code == 422
    assert service.list_documents() == []
    assert list(settings.upload_dir.iterdir()) == []
    assert vector_counts(service) == {}


def test_path_traversal_filename_is_neutralised(service, settings):
    record = service.upload("../../evil\\..\\name.pdf", LANGCHAIN_PDF)

    assert record["original_filename"] == "name.pdf"
    assert (settings.upload_dir / record["stored_filename"]).is_file()


# ---------------- delete ----------------

def test_delete_removes_file_registry_vectors_and_bm25(service, settings, pipeline):
    keep = service.upload("keep.pdf", LANGCHAIN_PDF)
    gone = service.upload("gone.pdf", GEMINI_PDF)

    service.delete(gone["document_id"])

    assert pdf_files(settings) == [keep["stored_filename"]]
    assert [d["document_id"] for d in service.list_documents()] == [keep["document_id"]]
    assert list(vector_counts(service)) == [keep["stored_filename"]]
    assert {c.metadata["source"] for c in pipeline.bm25_retriever.documents} == {
        keep["stored_filename"]
    }


def test_delete_unknown_document(service):
    with pytest.raises(DocumentNotFound):
        service.delete("does-not-exist")


def test_interrupted_delete_is_finished_by_reconcile(service, settings, monkeypatch):
    record = service.upload("a.pdf", LANGCHAIN_PDF)

    def explode(*_args, **_kwargs):
        raise RuntimeError("chroma went away")

    monkeypatch.setattr(ds, "delete_source", explode)

    with pytest.raises(RuntimeError):
        service.delete(record["document_id"])

    assert service.get_document(record["document_id"])["status"] == "deleting"
    with pytest.raises(DocumentUnavailable):  # not searchable while deleting
        service.retrieval_filters(None)

    monkeypatch.undo()
    report = service.reconcile()

    assert report["completed_deletions"] == ["a.pdf"]
    assert service.list_documents() == []
    assert pdf_files(settings) == []
    assert vector_counts(service) == {}


# ---------------- replace ----------------

def test_replace_swaps_content_and_keeps_document_id(service, settings, pipeline):
    old = service.upload("report.pdf", LANGCHAIN_PDF)

    result = service.replace(old["document_id"], "report-v2.pdf", GEMINI_PDF)
    new = result["document"]

    assert result["changed"] is True
    assert new["document_id"] == old["document_id"]
    assert new["created_at"] == old["created_at"]
    assert new["original_filename"] == "report-v2.pdf"
    assert new["stored_filename"] != old["stored_filename"]
    assert pdf_files(settings) == [new["stored_filename"]]
    assert list(vector_counts(service)) == [new["stored_filename"]]
    assert {c.metadata["source"] for c in pipeline.bm25_retriever.documents} == {
        new["stored_filename"]
    }


def test_replace_with_identical_bytes_is_a_noop(service):
    record = service.upload("a.pdf", LANGCHAIN_PDF)

    result = service.replace(record["document_id"], "a.pdf", LANGCHAIN_PDF)

    assert result["changed"] is False
    assert result["document"]["stored_filename"] == record["stored_filename"]


def test_replace_with_another_documents_bytes_is_rejected(service):
    first = service.upload("a.pdf", LANGCHAIN_PDF)
    second = service.upload("b.pdf", GEMINI_PDF)

    with pytest.raises(DuplicateDocument) as caught:
        service.replace(second["document_id"], "b.pdf", LANGCHAIN_PDF)

    assert caught.value.existing_document["document_id"] == first["document_id"]


def test_failed_replace_keeps_old_version_intact(service, settings, monkeypatch):
    old = service.upload("a.pdf", LANGCHAIN_PDF)

    def explode(*_args, **_kwargs):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(ds, "add_documents", explode)

    with pytest.raises(InvalidUpload):
        service.replace(old["document_id"], "a2.pdf", GEMINI_PDF)

    monkeypatch.undo()

    assert service.get_document(old["document_id"]) == old
    assert pdf_files(settings) == [old["stored_filename"]]
    assert vector_counts(service) == {old["stored_filename"]: old["chunk_count"]}


def test_replace_unknown_document(service):
    with pytest.raises(DocumentNotFound):
        service.replace("nope", "a.pdf", LANGCHAIN_PDF)


# ---------------- retrieval filters ----------------

def test_retrieval_filters(service):
    with pytest.raises(DocumentUnavailable) as empty:
        service.retrieval_filters(None)
    assert empty.value.status_code == 409

    record = service.upload("a.pdf", LANGCHAIN_PDF)

    assert service.retrieval_filters(None) == {"source": {"$in": [record["stored_filename"]]}}
    assert service.retrieval_filters(record["stored_filename"]) == {
        "source": record["stored_filename"]
    }

    with pytest.raises(DocumentUnavailable) as missing:
        service.retrieval_filters("deleted.pdf")
    assert missing.value.status_code == 404


def test_document_ids_select_indexed_documents(service):
    first = service.upload("a.pdf", LANGCHAIN_PDF)
    second = service.upload("b.pdf", GEMINI_PDF)

    assert service.retrieval_filters(document_ids=[first["document_id"]]) == {
        "source": first["stored_filename"]
    }
    assert service.retrieval_filters(
        document_ids=[second["document_id"], first["document_id"]]
    ) == {
        "source": {"$in": [second["stored_filename"], first["stored_filename"]]}
    }
    assert service.retrieval_filters(
        document_ids=[first["document_id"], f"  {first['document_id']}  "]
    ) == {"source": first["stored_filename"]}

    filters, labels = service.resolve_search(document_ids=[first["document_id"]])
    assert filters == {"source": first["stored_filename"]}
    assert labels == {
        first["stored_filename"]: {
            "document_id": first["document_id"],
            "original_filename": "a.pdf",
        }
    }


def test_document_ids_reject_empty_unknown_and_conflicting_selectors(service):
    first = service.upload("a.pdf", LANGCHAIN_PDF)
    second = service.upload("b.pdf", GEMINI_PDF)

    agreed = service.retrieval_filters(first["stored_filename"], [first["document_id"]])
    assert agreed == {"source": first["stored_filename"]}

    with pytest.raises(InvalidSearch):
        service.retrieval_filters(first["stored_filename"], [second["document_id"]])

    with pytest.raises(InvalidSearch):
        service.retrieval_filters(
            first["stored_filename"],
            [first["document_id"], second["document_id"]],
        )

    with pytest.raises(InvalidSearch):
        service.retrieval_filters(document_ids=[])

    with pytest.raises(InvalidSearch):
        service.retrieval_filters(first["stored_filename"], [])

    with pytest.raises(InvalidSearch):
        service.retrieval_filters(document_ids=[" "])

    with pytest.raises(DocumentUnavailable) as missing:
        service.retrieval_filters(document_ids=["missing-id"])
    assert missing.value.status_code == 404

    service.registry.update_status(second["document_id"], status="failed", error_message="stopped")

    with pytest.raises(DocumentUnavailable) as failed:
        service.retrieval_filters(document_ids=[second["document_id"]])
    assert failed.value.status_code == 404

    assert service.retrieval_filters(None) == {"source": {"$in": [first["stored_filename"]]}}


# ---------------- startup reconciliation ----------------

def new_service(settings, pipeline) -> DocumentService:
    """A fresh service over the same storage, as after a process restart."""

    return DocumentService(
        registry=DocumentRegistry(db_path=settings.documents_db),
        vector_store=pipeline.vector_store,
        settings=settings,
        set_bm25_documents=pipeline.set_bm25_documents,
    )


def test_restart_restores_bm25_without_touching_vectors(service, settings, pipeline):
    record = service.upload("a.pdf", LANGCHAIN_PDF)
    before = vector_counts(service)

    pipeline.set_bm25_documents([])  # in-memory state lost on restart
    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()

    assert report["failed"] == [] and report["reindexed"] == []
    assert vector_counts(restarted) == before
    assert len(pipeline.bm25_retriever.documents) == record["chunk_count"]


def test_untracked_pdf_is_adopted_and_indexed(service, settings):
    legacy = settings.upload_dir / ("0" * 32 + "_legacy.pdf")
    legacy.write_bytes(LANGCHAIN_PDF)

    report = service.reconcile()

    assert report["adopted_files"] == [legacy.name]
    [record] = service.list_documents()
    assert record["original_filename"] == "legacy.pdf"
    assert record["status"] == "indexed"
    assert vector_counts(service) == {legacy.name: record["chunk_count"]}


def test_sample_pdf_scenario_duplicate_file_and_stale_vectors(service, settings, pipeline):
    """Mirrors the real project: sample.pdf duplicates a registered PDF and
    ChromaDB holds vectors for it. Nothing may be deleted; retrieval must
    ignore the stale vectors."""

    record = service.upload("langchain_components.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / "sample.pdf").write_bytes(LANGCHAIN_PDF)  # identical bytes

    stale = [
        Document(
            page_content=f"stale chunk {i} about LangChain indexes retrievers",
            metadata={"source": "sample.pdf", "page": 1, "chunk_id": i},
        )
        for i in range(5)
    ]
    from backend.vector_store import add_documents

    add_documents(pipeline.vector_store, stale)

    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()

    assert report["duplicate_files"][0]["file"] == "sample.pdf"
    assert report["orphan_vectors"] == {"sample.pdf": 5}
    assert report["pruned_orphan_vectors"] == {}
    assert (settings.upload_dir / "sample.pdf").exists()  # file untouched
    assert vector_counts(restarted)["sample.pdf"] == 5  # vectors untouched
    assert len(restarted.list_documents()) == 1

    # ...but they can never be returned by a search.
    filters = restarted.retrieval_filters(None)
    hits = pipeline.hybrid_retriever.search("LangChain indexes", 5, 10, filters)
    assert hits
    assert {doc.metadata["source"] for doc, _ in hits} == {record["stored_filename"]}


def test_orphan_vectors_pruned_only_when_enabled(settings, pipeline):
    from backend.vector_store import add_documents

    add_documents(
        pipeline.vector_store,
        [Document(page_content="orphan", metadata={"source": "ghost.pdf", "page": 1, "chunk_id": 0})],
    )
    pruning = dataclasses.replace(settings, prune_orphan_vectors=True)

    report = new_service(pruning, pipeline).reconcile()

    assert report["pruned_orphan_vectors"] == {"ghost.pdf": 1}
    assert vector_counts(new_service(pruning, pipeline)) == {}


def test_interrupted_upload_is_recovered_or_marked_failed(service, settings):
    # 'processing' record whose file exists -> indexed on the next start.
    ok = service.registry.create_document("id-ok", "ok.pdf", "ok_stored.pdf", "h1", 10)
    (settings.upload_dir / "ok_stored.pdf").write_bytes(LANGCHAIN_PDF)
    # 'processing' record whose file never made it to disk -> failed.
    service.registry.create_document("id-gone", "gone.pdf", "gone_stored.pdf", "h2", 10)
    # Partially written temp file.
    (settings.upload_dir / "half.pdf.part").write_bytes(b"%PDF-1.4 trunc")

    report = service.reconcile()

    assert service.get_document("id-ok")["status"] == "indexed"
    assert service.get_document("id-gone")["status"] == "failed"
    assert "file is missing" in service.get_document("id-gone")["error_message"]
    assert report["removed_partial_files"] == ["half.pdf.part"]
    assert not (settings.upload_dir / "half.pdf.part").exists()


def test_reuploading_a_failed_document_works(service, settings):
    service.registry.create_document("id-1", "a.pdf", "a_stored.pdf", ds.sha256_bytes(LANGCHAIN_PDF), 10)
    service.reconcile()  # no file -> failed
    assert service.get_document("id-1")["status"] == "failed"

    record = service.upload("a.pdf", LANGCHAIN_PDF)

    assert record["status"] == "indexed"
    assert [d["document_id"] for d in service.list_documents()] == [record["document_id"]]


# ================================================================
# Replacement recovery (operation journal)
# ================================================================

def _fail_delete_source_for(monkeypatch, stored_name):
    real = ds.delete_source

    def selective(vector_store, source):
        if source == stored_name:
            raise RuntimeError("chroma unavailable")
        return real(vector_store, source)

    monkeypatch.setattr(ds, "delete_source", selective)


def test_replace_cleanup_failure_is_finished_on_restart(service, settings, pipeline, monkeypatch):
    old = service.upload("a.pdf", LANGCHAIN_PDF)
    _fail_delete_source_for(monkeypatch, old["stored_filename"])

    result = service.replace(old["document_id"], "b.pdf", GEMINI_PDF)
    new = result["document"]

    # The replacement itself succeeded; the old version is still on disk
    # and the journal says cleanup is pending.
    assert result["changed"] is True
    assert (settings.upload_dir / old["stored_filename"]).exists()
    [op] = service.registry.list_operations()
    assert op["state"] == "cleanup_pending"

    monkeypatch.undo()
    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()

    assert len(report["recovered_replacements"]) == 1
    assert report["adopted_files"] == [] and report["duplicate_files"] == []
    assert [d["document_id"] for d in restarted.list_documents()] == [old["document_id"]]
    assert restarted.get_document(old["document_id"])["stored_filename"] == new["stored_filename"]
    assert pdf_files(settings) == [new["stored_filename"]]
    assert list(vector_counts(restarted)) == [new["stored_filename"]]
    assert restarted.registry.list_operations() == []

    again = restarted.reconcile()  # idempotent
    assert again["recovered_replacements"] == [] and again["failed"] == []


def test_old_replaced_file_is_never_adopted_even_if_cleanup_keeps_failing(
    service, settings, pipeline, monkeypatch
):
    old = service.upload("a.pdf", LANGCHAIN_PDF)
    _fail_delete_source_for(monkeypatch, old["stored_filename"])
    service.replace(old["document_id"], "b.pdf", GEMINI_PDF)

    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()  # still failing

    assert report["pending_cleanups"]
    assert report["adopted_files"] == []
    assert len(restarted.list_documents()) == 1
    assert restarted.registry.list_operations() != []  # kept for the next attempt
    assert restarted.get_document(old["document_id"])["status"] == "indexed"


def test_crash_after_registry_switch_keeps_new_version(service, settings, pipeline, monkeypatch):
    old = service.upload("a.pdf", LANGCHAIN_PDF)

    def crash(*_args, **_kwargs):
        raise RuntimeError("process killed")

    monkeypatch.setattr(service, "_finish_replace_cleanup", crash)
    new = service.replace(old["document_id"], "b.pdf", GEMINI_PDF)["document"]
    monkeypatch.undo()

    restarted = new_service(settings, pipeline)
    restarted.reconcile()

    assert restarted.get_document(old["document_id"])["stored_filename"] == new["stored_filename"]
    assert pdf_files(settings) == [new["stored_filename"]]
    assert restarted.registry.list_operations() == []


def test_crash_before_registry_switch_keeps_old_version(service, settings, pipeline):
    """Hard crash (no exception handling) after the new version was built."""

    old = service.upload("a.pdf", LANGCHAIN_PDF)

    new_stored = "f" * 32 + "_b.pdf"
    (settings.upload_dir / new_stored).write_bytes(GEMINI_PDF)
    _pages, chunks = service._extract_and_chunk(settings.upload_dir / new_stored)
    from backend.vector_store import add_documents

    add_documents(pipeline.vector_store, chunks)
    service.registry.begin_replace_operation(
        "op-1", old["document_id"], old["stored_filename"], new_stored
    )

    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()

    assert restarted.get_document(old["document_id"]) == old
    assert pdf_files(settings) == [old["stored_filename"]]
    assert list(vector_counts(restarted)) == [old["stored_filename"]]
    assert report["adopted_files"] == []
    assert restarted.registry.list_operations() == []


def test_failed_replace_leaves_no_journal_entry(service, monkeypatch):
    old = service.upload("a.pdf", LANGCHAIN_PDF)
    monkeypatch.setattr(
        service.registry, "replace_file", lambda **_: (_ for _ in ()).throw(RuntimeError("db"))
    )

    with pytest.raises(InvalidUpload):
        service.replace(old["document_id"], "b.pdf", GEMINI_PDF)

    assert service.registry.list_operations() == []
    assert service.get_document(old["document_id"]) == old


# ================================================================
# Deletion recovery at every step
# ================================================================

def _break_unlink(monkeypatch, stored_name):
    from pathlib import Path

    real = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == stored_name:
            raise PermissionError("file is locked")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)


@pytest.mark.parametrize("failing_step", ["vectors", "file", "registry"])
def test_deletion_interrupted_at_each_step_recovers(
    service, settings, pipeline, monkeypatch, failing_step
):
    keep = service.upload("keep.pdf", GEMINI_PDF)
    target = service.upload("target.pdf", LANGCHAIN_PDF)

    if failing_step == "vectors":
        _fail_delete_source_for(monkeypatch, target["stored_filename"])
    elif failing_step == "file":
        _break_unlink(monkeypatch, target["stored_filename"])
    else:
        monkeypatch.setattr(
            service.registry, "delete_document",
            lambda *_: (_ for _ in ()).throw(RuntimeError("db locked")),
        )

    with pytest.raises(Exception):
        service.delete(target["document_id"])

    assert service.get_document(target["document_id"])["status"] == "deleting"
    assert service.retrieval_filters(None) == {"source": {"$in": [keep["stored_filename"]]}}

    monkeypatch.undo()
    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()

    assert report["completed_deletions"] == ["target.pdf"]
    assert [d["document_id"] for d in restarted.list_documents()] == [keep["document_id"]]
    assert pdf_files(settings) == [keep["stored_filename"]]
    assert list(vector_counts(restarted)) == [keep["stored_filename"]]
    assert {c.metadata["source"] for c in pipeline.bm25_retriever.documents} == {keep["stored_filename"]}

    assert restarted.reconcile()["completed_deletions"] == []  # repeatable


def test_deletion_that_cannot_finish_is_never_resurrected(service, settings, pipeline, monkeypatch):
    record = service.upload("a.pdf", LANGCHAIN_PDF)
    _fail_delete_source_for(monkeypatch, record["stored_filename"])

    with pytest.raises(RuntimeError):
        service.delete(record["document_id"])

    restarted = new_service(settings, pipeline)
    restarted.reconcile()  # delete still failing

    assert restarted.get_document(record["document_id"])["status"] == "deleting"
    assert pipeline.bm25_retriever is None  # not re-indexed
    with pytest.raises(DocumentUnavailable):
        restarted.retrieval_filters(None)

    monkeypatch.undo()
    restarted.delete(record["document_id"])  # user can simply retry the delete
    assert restarted.list_documents() == []


def test_delete_tolerates_missing_file_and_missing_vectors(service, settings):
    from backend.vector_store import delete_source

    record = service.upload("a.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / record["stored_filename"]).unlink()
    delete_source(service.vector_store, record["stored_filename"])

    service.delete(record["document_id"])

    assert service.list_documents() == []
    with pytest.raises(DocumentNotFound):  # repeated deletion is a clean 404
        service.delete(record["document_id"])


# ================================================================
# Reconciliation matrix: remaining cases
# ================================================================

def test_registered_pdf_without_vectors_is_reindexed(service, settings, pipeline):
    from backend.vector_store import delete_source

    record = service.upload("a.pdf", LANGCHAIN_PDF)
    delete_source(pipeline.vector_store, record["stored_filename"])
    assert vector_counts(service) == {}

    restarted = new_service(settings, pipeline)
    restarted.reconcile()

    assert vector_counts(restarted) == {record["stored_filename"]: record["chunk_count"]}
    assert restarted.get_document(record["document_id"])["status"] == "indexed"


def test_registered_pdf_missing_is_marked_failed_and_not_searched(service, settings, pipeline):
    record = service.upload("a.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / record["stored_filename"]).unlink()

    restarted = new_service(settings, pipeline)
    report = restarted.reconcile()

    assert restarted.get_document(record["document_id"])["status"] == "failed"
    assert report["failed"][0]["reason"] == "file missing"
    with pytest.raises(DocumentUnavailable):
        restarted.retrieval_filters(None)


def test_reconcile_is_repeatable_on_a_messy_state(service, settings, pipeline):
    service.upload("a.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / "sample.pdf").write_bytes(LANGCHAIN_PDF)
    (settings.upload_dir / "x.pdf.part").write_bytes(b"%PDF-")
    (settings.upload_dir / ("1" * 32 + "_new.pdf")).write_bytes(GEMINI_PDF)

    first = new_service(settings, pipeline).reconcile()
    second = new_service(settings, pipeline).reconcile()

    assert first["adopted_files"] and first["duplicate_files"]
    assert second["adopted_files"] == [] and second["removed_partial_files"] == []
    assert len(new_service(settings, pipeline).list_documents()) == 2


# ================================================================
# Explicit cleanup of duplicates / stale vectors
# ================================================================

def _sample_pdf_state(service, settings, pipeline):
    from backend.vector_store import add_documents

    record = service.upload("langchain_components.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / "sample.pdf").write_bytes(LANGCHAIN_PDF)
    add_documents(
        pipeline.vector_store,
        [
            Document(page_content=f"stale {i}", metadata={"source": "sample.pdf", "page": 1, "chunk_id": i})
            for i in range(3)
        ],
    )
    return record


def test_cleanup_plan_is_read_only_and_explains_evidence(service, settings, pipeline):
    record = _sample_pdf_state(service, settings, pipeline)

    plan = service.plan_cleanup()

    assert [f["file"] for f in plan["duplicate_files"]] == ["sample.pdf"]
    assert plan["duplicate_files"][0]["duplicate_of"]["document_id"] == record["document_id"]
    assert plan["orphan_vectors"] == [
        {"source": "sample.pdf", "vector_count": 3, "metadata_verified": True, "duplicate_file_on_disk": True}
    ]
    assert (settings.upload_dir / "sample.pdf").exists()
    assert vector_counts(service)["sample.pdf"] == 3


def test_cleanup_apply_quarantines_instead_of_deleting(service, settings, pipeline):
    import json

    record = _sample_pdf_state(service, settings, pipeline)

    manifest = service.apply_cleanup()

    assert not (settings.upload_dir / "sample.pdf").exists()
    quarantined = settings.data_dir / "quarantine"
    [stamp_dir] = list(quarantined.iterdir())
    assert (stamp_dir / "files" / "sample.pdf").read_bytes() == LANGCHAIN_PDF
    backup = json.loads(next((stamp_dir / "vectors").iterdir()).read_text(encoding="utf-8"))
    assert backup["source"] == "sample.pdf" and len(backup["ids"]) == 3
    assert manifest["vectors"][0]["removed"] == 3

    # The registered document is untouched.
    assert vector_counts(service) == {record["stored_filename"]: record["chunk_count"]}
    assert (settings.upload_dir / record["stored_filename"]).exists()
    assert service.plan_cleanup() == {"duplicate_files": [], "orphan_vectors": []}


def test_cleanup_never_touches_unknown_files_or_registered_sources(service, settings, pipeline):
    record = service.upload("a.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / "unknown.pdf").write_bytes(GEMINI_PDF)  # not a duplicate of anything

    plan = service.plan_cleanup()

    assert plan == {"duplicate_files": [], "orphan_vectors": []}
    service.apply_cleanup()
    assert (settings.upload_dir / "unknown.pdf").exists()
    assert vector_counts(service) == {record["stored_filename"]: record["chunk_count"]}


def test_cleanup_keeps_duplicate_if_registered_copy_is_missing(service, settings):
    record = service.upload("a.pdf", LANGCHAIN_PDF)
    (settings.upload_dir / "copy.pdf").write_bytes(LANGCHAIN_PDF)
    (settings.upload_dir / record["stored_filename"]).unlink()

    assert service.plan_cleanup()["duplicate_files"] == []
