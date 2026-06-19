from __future__ import annotations

from advanced_rag.models import Chunk, SearchResult


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "rag_chunks") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed") from exc
        client = chromadb.PersistentClient(path=persist_directory)
        self.collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings[start : start + batch_size],
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "parent_id": chunk.parent_id,
                        "filename": chunk.filename,
                        "heading": chunk.heading,
                        "page": chunk.page or 0,
                        "position": chunk.position,
                    }
                    for chunk in batch
                ],
            )

    def search(self, embedding: list[float], limit: int = 40) -> list[SearchResult]:
        response = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        output: list[SearchResult] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            metadata = metadata or {}
            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    text=text or "",
                    filename=str(metadata.get("filename", "")),
                    heading=str(metadata.get("heading", "")),
                    page=int(metadata.get("page", 0)) or None,
                    parent_id=str(metadata.get("parent_id", "")),
                    score=1.0 - float(distance),
                )
            )
        return output

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
