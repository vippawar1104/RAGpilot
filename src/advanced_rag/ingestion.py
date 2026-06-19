from __future__ import annotations

import hashlib
import mimetypes
import uuid
from pathlib import Path

from advanced_rag.chunking import HierarchicalChunker
from advanced_rag.config import Settings
from advanced_rag.database import Database
from advanced_rag.embeddings import LocalEmbedder
from advanced_rag.models import DocumentRecord, DocumentStatus
from advanced_rag.parser import DocumentParser
from advanced_rag.vector_store import ChromaVectorStore


class IngestionService:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self.parser = DocumentParser()
        self.embedder = LocalEmbedder(settings.embedding_model, settings.device)
        self.vector_store = ChromaVectorStore(str(settings.chroma_dir))

    def enqueue_bytes(self, filename: str, content: bytes) -> tuple[DocumentRecord, bool]:
        checksum = hashlib.sha256(content).hexdigest()
        if existing := self.database.find_by_checksum(checksum):
            return existing, True

        document_id = uuid.uuid4().hex
        safe_name = Path(filename).name.replace("\x00", "")
        destination = self.settings.upload_dir / f"{document_id}_{safe_name}"
        destination.write_bytes(content)
        record = DocumentRecord(
            id=document_id,
            filename=safe_name,
            path=destination,
            checksum=checksum,
            status=DocumentStatus.QUEUED,
            size_bytes=len(content),
            mime_type=mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
        )
        self.database.add_document(record)
        return record, False

    def enqueue_file(self, source: Path) -> tuple[DocumentRecord, bool]:
        return self.enqueue_bytes(source.name, source.read_bytes())

    def process_document(self, document_id: str) -> int:
        document = self.database.get_document(document_id)
        if not document:
            raise ValueError(f"Unknown document: {document_id}")

        parsed_path = self.settings.parsed_dir / f"{document_id}.json"
        blocks = self.parser.parse(document.path, parsed_path)
        chunker = HierarchicalChunker(
            child_tokens=self.settings.child_chunk_tokens,
            parent_tokens=self.settings.parent_chunk_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        chunks = chunker.chunk(document.id, document.filename, blocks)
        if not chunks:
            raise ValueError("No usable text was extracted from the document")

        embeddings = self.embedder.encode_documents([chunk.text for chunk in chunks])

        # Chroma first: a DB failure leaves harmless vectors that a retry will upsert.
        self.vector_store.delete_document(document_id)
        self.vector_store.upsert(chunks, embeddings)
        self.database.replace_chunks(document_id, chunks)
        self.database.finish_job(document_id, len(chunks))
        return len(chunks)

    def process_next(self) -> str | None:
        document_id = self.database.claim_next_job()
        if not document_id:
            return None
        try:
            self.process_document(document_id)
        except Exception as exc:
            self.database.fail_job(document_id, str(exc))
        return document_id

    def delete_document(self, document_id: str) -> None:
        document = self.database.get_document(document_id)
        if not document:
            return
        self.vector_store.delete_document(document_id)
        self.database.delete_document_rows(document_id)
        document.path.unlink(missing_ok=True)
        (self.settings.parsed_dir / f"{document_id}.json").unlink(missing_ok=True)
