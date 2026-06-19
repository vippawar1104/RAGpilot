from __future__ import annotations

import time
from collections import defaultdict

from advanced_rag.config import Settings
from advanced_rag.database import Database
from advanced_rag.embeddings import LocalEmbedder, LocalReranker
from advanced_rag.models import RetrievalTrace, SearchResult
from advanced_rag.vector_store import ChromaVectorStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]], constant: int = 60
) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    results: dict[str, SearchResult] = {}
    for list_index, ranked in enumerate(ranked_lists):
        for rank, result in enumerate(ranked, start=1):
            scores[result.chunk_id] += 1.0 / (constant + rank)
            results.setdefault(result.chunk_id, result)
            if list_index == 0:
                results[result.chunk_id].dense_rank = rank
            elif list_index == 1:
                results[result.chunk_id].lexical_rank = rank
    for chunk_id, score in scores.items():
        results[chunk_id].score = score
    return sorted(results.values(), key=lambda item: item.score, reverse=True)


class RetrievalEngine:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self.embedder = LocalEmbedder(settings.embedding_model, settings.device)
        self.reranker = LocalReranker(settings.reranker_model, settings.device)
        self.vector_store = ChromaVectorStore(str(settings.chroma_dir))

    def retrieve(self, query: str) -> RetrievalTrace:
        query = " ".join(query.split()).strip()
        if not query:
            raise ValueError("Query cannot be empty")
        timings: dict[str, float] = {}

        started = time.perf_counter()
        query_embedding = self.embedder.encode_query(query)
        dense = self.vector_store.search(query_embedding, self.settings.dense_top_k)
        timings["dense"] = self._elapsed_ms(started)

        started = time.perf_counter()
        lexical = self.database.lexical_search(query, self.settings.lexical_top_k)
        timings["lexical"] = self._elapsed_ms(started)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion([dense, lexical])
        candidates = fused[: self.settings.rerank_top_k]
        timings["fusion"] = self._elapsed_ms(started)

        started = time.perf_counter()
        try:
            scores = self.reranker.score(query, [candidate.text for candidate in candidates])
            for candidate, score in zip(candidates, scores, strict=False):
                candidate.reranker_score = score
            candidates.sort(
                key=lambda item: (
                    item.reranker_score if item.reranker_score is not None else float("-inf")
                ),
                reverse=True,
            )
        except Exception:
            # Retrieval remains useful when the optional reranker cannot load.
            pass
        timings["rerank"] = self._elapsed_ms(started)

        final = self._select_diverse(candidates, self.settings.final_top_k)
        timings["total"] = sum(timings.values())
        return RetrievalTrace(
            query=query,
            dense=dense,
            lexical=lexical,
            fused=fused,
            final=final,
            timings_ms=timings,
        )

    def build_context(self, results: list[SearchResult]) -> tuple[str, list[SearchResult]]:
        sections: list[str] = []
        included: list[SearchResult] = []
        used_chars = 0
        seen_parents: set[str] = set()
        for index, result in enumerate(results, start=1):
            if result.parent_id in seen_parents:
                text = result.text
            else:
                parent_text = self.database.get_parent_text(result.parent_id)
                text = parent_text or result.text
                seen_parents.add(result.parent_id)
            header = f"[S{index}] {result.citation_label}"
            if result.heading:
                header += f" | {result.heading}"
            section = f"{header}\n{text.strip()}"
            if sections and used_chars + len(section) > self.settings.max_context_chars:
                break
            sections.append(section)
            included.append(result)
            used_chars += len(section)
        return "\n\n---\n\n".join(sections), included

    @staticmethod
    def _select_diverse(candidates: list[SearchResult], limit: int) -> list[SearchResult]:
        selected: list[SearchResult] = []
        parent_counts: dict[str, int] = defaultdict(int)
        document_counts: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            if parent_counts[candidate.parent_id] >= 1:
                continue
            if document_counts[candidate.document_id] >= 4:
                continue
            selected.append(candidate)
            parent_counts[candidate.parent_id] += 1
            document_counts[candidate.document_id] += 1
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
