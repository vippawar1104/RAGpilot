from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

from advanced_rag.models import SearchResult

SYSTEM_PROMPT = """You are a precise document research assistant.
Answer using only the supplied sources. Every factual claim must include one or more source
markers such as [S1] or [S2]. If the sources do not support an answer, state that the available
documents are insufficient. Do not invent facts, filenames, page numbers, or source markers.
When sources conflict, describe the conflict and cite both. Prefer a direct answer over a long
preface."""


@dataclass(slots=True)
class GenerationConfig:
    api_key: str
    model: str
    provider: str = "auto"
    base_url: str | None = None
    temperature: float = 0.1
    max_output_tokens: int = 4096


def resolve_provider(provider: str, api_key: str) -> str:
    normalized = provider.strip().lower()
    if normalized != "auto":
        if normalized not in {"anthropic", "openai"}:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        return normalized
    return "anthropic" if api_key.startswith("sk-ant-") else "openai"


class AnswerGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        if not config.api_key:
            raise ValueError("An LLM API key is required")
        self.config = config
        self.provider = resolve_provider(config.provider, config.api_key)
        self.client = self._create_client()

    def _create_client(self):
        if self.provider == "anthropic":
            try:
                import httpx
            except ImportError as exc:
                raise RuntimeError("The httpx package is not installed") from exc
            return httpx.Client(timeout=120.0)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed") from exc
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return OpenAI(**kwargs)

    def stream(
        self,
        question: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        recent_history = (history or [])[-6:]
        messages = list(
            message
            for message in recent_history
            if message.get("role") in {"user", "assistant"} and message.get("content")
        )
        messages.append(
            {
                "role": "user",
                "content": f"SOURCES:\n{context}\n\nQUESTION:\n{question}",
            }
        )
        if self.provider == "anthropic":
            yield from self._stream_anthropic(messages)
            return

        openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            temperature=self.config.temperature,
            stream=True,
        )
        for event in stream:
            content = event.choices[0].delta.content if event.choices else None
            if content:
                yield content

    def _stream_anthropic(self, messages: list[dict[str, str]]) -> Iterator[str]:
        base_url = (self.config.base_url or "https://api.anthropic.com").rstrip("/")
        url = base_url if base_url.endswith("/v1/messages") else f"{base_url}/v1/messages"
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": self.config.api_key,
        }
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": True,
        }
        with self.client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("type") != "content_block_delta":
                    continue
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]


def source_summary(sources: list[SearchResult]) -> list[dict[str, str | int | None | float]]:
    return [
        {
            "source": f"S{index}",
            "filename": source.filename,
            "page": source.page,
            "heading": source.heading,
            "score": source.reranker_score if source.reranker_score is not None else source.score,
            "excerpt": source.text[:500],
        }
        for index, source in enumerate(sources, start=1)
    ]
