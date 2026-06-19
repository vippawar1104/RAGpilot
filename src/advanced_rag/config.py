from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore", case_sensitive=False
    )

    data_dir: Path = Path("data")
    llm_api_key: str = ""
    llm_provider: str = "auto"
    llm_model: str = "claude-sonnet-4-6"
    llm_base_url: str | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"

    child_chunk_tokens: int = Field(default=420, ge=100, le=1200)
    parent_chunk_tokens: int = Field(default=1500, ge=400, le=5000)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=300)
    dense_top_k: int = Field(default=40, ge=1, le=200)
    lexical_top_k: int = Field(default=40, ge=1, le=200)
    rerank_top_k: int = Field(default=30, ge=1, le=100)
    final_top_k: int = Field(default=10, ge=1, le=30)
    max_context_chars: int = Field(default=28000, ge=2000, le=100000)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.parsed_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
