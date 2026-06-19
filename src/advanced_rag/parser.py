from __future__ import annotations

import json
import re
from pathlib import Path

from advanced_rag.models import ParsedBlock

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}


class DocumentParser:
    def parse(self, path: Path, parsed_output: Path | None = None) -> list[ParsedBlock]:
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            blocks = self._blocks_from_markdown(text)
        else:
            blocks, text = self._parse_with_docling(path)

        if parsed_output:
            parsed_output.parent.mkdir(parents=True, exist_ok=True)
            parsed_output.write_text(
                json.dumps(
                    [
                        {"text": block.text, "heading": block.heading, "page": block.page}
                        for block in blocks
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return blocks

    def _parse_with_docling(self, path: Path) -> tuple[list[ParsedBlock], str]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                f"Docling is required to parse {path.suffix} files. Install project dependencies."
            ) from exc

        result = DocumentConverter().convert(path)
        markdown = result.document.export_to_markdown()
        return self._blocks_from_markdown(markdown), markdown

    @staticmethod
    def _blocks_from_markdown(text: str) -> list[ParsedBlock]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        heading = ""
        blocks: list[ParsedBlock] = []
        buffer: list[str] = []

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                content = re.sub(r"\n{3,}", "\n\n", content)
                blocks.append(ParsedBlock(text=content, heading=heading))
            buffer.clear()

        for line in text.splitlines():
            if match := re.match(r"^(#{1,6})\s+(.+?)\s*$", line):
                flush()
                heading = match.group(2).strip()
                continue
            if not line.strip() and buffer:
                flush()
            elif line.strip():
                buffer.append(line.rstrip())
        flush()
        return blocks or [ParsedBlock(text=text.strip())]
