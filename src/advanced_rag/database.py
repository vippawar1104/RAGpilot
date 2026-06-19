from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from advanced_rag.models import Chunk, DocumentRecord, DocumentStatus, SearchResult

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mime_type TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id TEXT NOT NULL,
    text TEXT NOT NULL,
    filename TEXT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    page INTEGER,
    position INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    filename,
    heading,
    tokenize='porter unicode61'
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_document(self, document: DocumentRecord) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO documents (
                   id, filename, path, checksum, status, size_bytes, mime_type,
                   created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.id,
                    document.filename,
                    str(document.path),
                    document.checksum,
                    document.status,
                    document.size_bytes,
                    document.mime_type,
                    now,
                    now,
                ),
            )
            connection.execute(
                """INSERT INTO jobs (document_id, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (document.id, DocumentStatus.QUEUED, now, now),
            )

    def find_by_checksum(self, checksum: str) -> DocumentRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE checksum = ?", (checksum,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._document_from_row(row) for row in rows]

    def claim_next_job(self) -> str | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT document_id FROM jobs WHERE status = ? ORDER BY id LIMIT 1",
                (DocumentStatus.QUEUED,),
            ).fetchone()
            if not row:
                return None
            document_id = str(row["document_id"])
            now = _now()
            connection.execute(
                """UPDATE jobs SET status = ?, attempts = attempts + 1, updated_at = ?
                   WHERE document_id = ?""",
                (DocumentStatus.PROCESSING, now, document_id),
            )
            connection.execute(
                "UPDATE documents SET status = ?, error = '', updated_at = ? WHERE id = ?",
                (DocumentStatus.PROCESSING, now, document_id),
            )
            return document_id

    def finish_job(self, document_id: str, chunk_count: int) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = '', updated_at = ? WHERE document_id = ?",
                (DocumentStatus.READY, now, document_id),
            )
            connection.execute(
                """UPDATE documents SET status = ?, error = '', chunk_count = ?, updated_at = ?
                   WHERE id = ?""",
                (DocumentStatus.READY, chunk_count, now, document_id),
            )

    def fail_job(self, document_id: str, error: str) -> None:
        now = _now()
        message = error[:2000]
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE document_id = ?",
                (DocumentStatus.FAILED, message, now, document_id),
            )
            connection.execute(
                "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (DocumentStatus.FAILED, message, now, document_id),
            )

    def retry_document(self, document_id: str) -> None:
        now = _now()
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = '', updated_at = ? WHERE document_id = ?",
                (DocumentStatus.QUEUED, now, document_id),
            )
            connection.execute(
                "UPDATE documents SET status = ?, error = '', updated_at = ? WHERE id = ?",
                (DocumentStatus.QUEUED, now, document_id),
            )

    def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        with self.connection() as connection:
            old_ids = connection.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", [(row["id"],) for row in old_ids]
            )
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.executemany(
                """INSERT INTO chunks
                   (id, document_id, parent_id, text, filename, heading, page, position,
                    token_count, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.parent_id,
                        chunk.text,
                        chunk.filename,
                        chunk.heading,
                        chunk.page,
                        chunk.position,
                        chunk.token_count,
                        json.dumps(chunk.metadata),
                    )
                    for chunk in chunks
                ],
            )
            connection.executemany(
                "INSERT INTO chunks_fts (chunk_id, text, filename, heading) VALUES (?, ?, ?, ?)",
                [(chunk.id, chunk.text, chunk.filename, chunk.heading) for chunk in chunks],
            )

    def lexical_search(self, query: str, limit: int = 40) -> list[SearchResult]:
        terms = [term.replace('"', "") for term in query.split() if len(term) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:20])
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT c.*, bm25(chunks_fts, 0.0, 1.0, 2.0, 1.5) AS bm25_score
                   FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id
                   WHERE chunks_fts MATCH ? ORDER BY bm25_score LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        return [self._search_result_from_row(row) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, SearchResult]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
            ).fetchall()
        return {row["id"]: self._search_result_from_row(row) for row in rows}

    def get_parent_text(self, parent_id: str) -> str:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT text FROM chunks WHERE parent_id = ? ORDER BY position", (parent_id,)
            ).fetchall()
        return "\n\n".join(row["text"] for row in rows)

    def delete_document_rows(self, document_id: str) -> None:
        with self.connection() as connection:
            ids = connection.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", [(row["id"],) for row in ids]
            )
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            path=Path(row["path"]),
            checksum=row["checksum"],
            status=row["status"],
            size_bytes=row["size_bytes"],
            mime_type=row["mime_type"],
            error=row["error"],
            chunk_count=row["chunk_count"],
        )

    @staticmethod
    def _search_result_from_row(row: sqlite3.Row) -> SearchResult:
        return SearchResult(
            chunk_id=row["id"],
            document_id=row["document_id"],
            text=row["text"],
            filename=row["filename"],
            heading=row["heading"],
            page=row["page"],
            parent_id=row["parent_id"],
        )
