from pathlib import Path

from advanced_rag.database import Database
from advanced_rag.models import Chunk, DocumentRecord, DocumentStatus


def test_document_queue_and_lexical_search(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    document = DocumentRecord(
        id="doc-1",
        filename="policy.md",
        path=tmp_path / "policy.md",
        checksum="abc",
        status=DocumentStatus.QUEUED,
        size_bytes=100,
    )
    database.add_document(document)
    assert database.claim_next_job() == "doc-1"

    chunks = [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            parent_id="parent-1",
            text="The cancellation policy allows refunds within thirty days.",
            filename="policy.md",
            heading="Refunds",
            position=0,
            token_count=9,
        )
    ]
    database.replace_chunks("doc-1", chunks)
    database.finish_job("doc-1", 1)

    results = database.lexical_search("cancellation refund", 10)
    assert results[0].chunk_id == "chunk-1"
    assert database.get_document("doc-1").status == DocumentStatus.READY


def test_duplicate_checksum_lookup(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    document = DocumentRecord(
        id="doc-1",
        filename="a.txt",
        path=tmp_path / "a.txt",
        checksum="same-content",
        status=DocumentStatus.QUEUED,
        size_bytes=10,
    )
    database.add_document(document)
    assert database.find_by_checksum("same-content").id == "doc-1"
