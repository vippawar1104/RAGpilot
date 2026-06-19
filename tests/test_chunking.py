from advanced_rag.chunking import HierarchicalChunker, estimate_tokens
from advanced_rag.models import ParsedBlock


def test_chunker_creates_stable_hierarchy_and_respects_order():
    blocks = [
        ParsedBlock(text="Alpha sentence. " * 15, heading="First"),
        ParsedBlock(text="Beta sentence. " * 15, heading="Second"),
        ParsedBlock(text="Gamma sentence. " * 15, heading="Third"),
    ]
    chunker = HierarchicalChunker(child_tokens=40, parent_tokens=100, overlap_tokens=5)

    first = chunker.chunk("doc-1", "sample.md", blocks)
    second = chunker.chunk("doc-1", "sample.md", blocks)

    assert first
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.position for chunk in first] == list(range(len(first)))
    assert all(chunk.document_id == "doc-1" for chunk in first)
    assert all(chunk.parent_id for chunk in first)


def test_token_estimate_handles_empty_and_punctuation():
    assert estimate_tokens("") == 1
    assert estimate_tokens("hello, world!") == 4
