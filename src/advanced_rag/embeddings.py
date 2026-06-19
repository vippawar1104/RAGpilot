from __future__ import annotations

from threading import Lock


class LocalEmbedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError("sentence-transformers is not installed") from exc
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        return self.encode_documents([query])[0]


class LocalReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as exc:
                        raise RuntimeError("sentence-transformers is not installed") from exc
                    self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict([(query, text) for text in texts], show_progress_bar=False)
        return [float(score) for score in scores]
