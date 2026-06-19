from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class ParsedBlock:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    parent_id: str
    text: str
    filename: str
    heading: str = ""
    page: int | None = None
    position: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentRecord:
    id: str
    filename: str
    path: Path
    checksum: str
    status: str
    size_bytes: int
    mime_type: str = ""
    error: str = ""
    chunk_count: int = 0


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    filename: str
    heading: str
    page: int | None
    parent_id: str
    score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    reranker_score: float | None = None

    @property
    def citation_label(self) -> str:
        location = f", p. {self.page}" if self.page else ""
        return f"{self.filename}{location}"


@dataclass(slots=True)
class RetrievalTrace:
    query: str
    dense: list[SearchResult]
    lexical: list[SearchResult]
    fused: list[SearchResult]
    final: list[SearchResult]
    timings_ms: dict[str, float] = field(default_factory=dict)
