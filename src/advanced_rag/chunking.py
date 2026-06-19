from __future__ import annotations

import hashlib
import re

from advanced_rag.models import Chunk, ParsedBlock


def estimate_tokens(text: str) -> int:
    # Stable model-independent approximation suitable for chunk boundaries.
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


class HierarchicalChunker:
    def __init__(
        self,
        child_tokens: int = 420,
        parent_tokens: int = 1500,
        overlap_tokens: int = 60,
    ) -> None:
        if child_tokens >= parent_tokens:
            raise ValueError("child_tokens must be smaller than parent_tokens")
        self.child_tokens = child_tokens
        self.parent_tokens = parent_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document_id: str, filename: str, blocks: list[ParsedBlock]) -> list[Chunk]:
        units = self._split_oversized_blocks(blocks)
        parents = self._group(units, self.parent_tokens)
        chunks: list[Chunk] = []
        position = 0

        for parent_index, parent_blocks in enumerate(parents):
            parent_id = _stable_id(document_id, "parent", str(parent_index))
            children = self._group(parent_blocks, self.child_tokens)
            previous_tail = ""
            for child_index, child_blocks in enumerate(children):
                body = "\n\n".join(block.text.strip() for block in child_blocks).strip()
                if previous_tail and child_index > 0:
                    body = f"{previous_tail}\n\n{body}"
                heading = next((block.heading for block in child_blocks if block.heading), "")
                page = next((block.page for block in child_blocks if block.page), None)
                chunk_id = _stable_id(document_id, parent_id, str(child_index), body)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        parent_id=parent_id,
                        text=body,
                        filename=filename,
                        heading=heading,
                        page=page,
                        position=position,
                        token_count=estimate_tokens(body),
                        metadata={"parent_index": parent_index, "child_index": child_index},
                    )
                )
                position += 1
                previous_tail = self._tail(body)
        return chunks

    def _split_oversized_blocks(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        output: list[ParsedBlock] = []
        for block in blocks:
            if estimate_tokens(block.text) <= self.child_tokens:
                output.append(block)
                continue
            sentences = re.split(r"(?<=[.!?])\s+|\n+", block.text)
            current: list[str] = []
            current_tokens = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                tokens = estimate_tokens(sentence)
                if current and current_tokens + tokens > self.child_tokens:
                    output.append(
                        ParsedBlock(" ".join(current), heading=block.heading, page=block.page)
                    )
                    current, current_tokens = [], 0
                if tokens > self.child_tokens:
                    words = sentence.split()
                    for start in range(0, len(words), self.child_tokens):
                        output.append(
                            ParsedBlock(
                                " ".join(words[start : start + self.child_tokens]),
                                heading=block.heading,
                                page=block.page,
                            )
                        )
                else:
                    current.append(sentence)
                    current_tokens += tokens
            if current:
                output.append(
                    ParsedBlock(" ".join(current), heading=block.heading, page=block.page)
                )
        return output

    @staticmethod
    def _group(blocks: list[ParsedBlock], max_tokens: int) -> list[list[ParsedBlock]]:
        groups: list[list[ParsedBlock]] = []
        current: list[ParsedBlock] = []
        count = 0
        for block in blocks:
            tokens = estimate_tokens(block.text)
            if current and count + tokens > max_tokens:
                groups.append(current)
                current, count = [], 0
            current.append(block)
            count += tokens
        if current:
            groups.append(current)
        return groups

    def _tail(self, text: str) -> str:
        if self.overlap_tokens == 0:
            return ""
        words = text.split()
        return " ".join(words[-self.overlap_tokens :])
