from advanced_rag.generation import resolve_provider
from advanced_rag.models import SearchResult
from advanced_rag.retrieval import reciprocal_rank_fusion


def result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc",
        text=chunk_id,
        filename="test.md",
        heading="",
        page=None,
        parent_id=f"parent-{chunk_id}",
    )


def test_rrf_rewards_candidates_found_by_both_retrievers():
    dense = [result("dense-only"), result("shared")]
    lexical = [result("shared"), result("lexical-only")]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0].chunk_id == "shared"
    assert fused[0].dense_rank == 2
    assert fused[0].lexical_rank == 1


def test_provider_is_detected_from_key_prefix():
    assert resolve_provider("auto", "sk-ant-example") == "anthropic"
    assert resolve_provider("auto", "sk-example") == "openai"
    assert resolve_provider("anthropic", "anything") == "anthropic"
