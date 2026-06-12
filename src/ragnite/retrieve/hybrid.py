"""Result fusion. Reciprocal Rank Fusion is rank-based, so dense cosine
scores and BM25 scores combine without any normalization step."""

from __future__ import annotations

from ragnite.types import ScoredChunk


def rrf_fuse(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists: score(chunk) = sum_i weight_i / (k + rank_i)."""
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    fused: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    for results, weight in zip(result_lists, weights, strict=True):
        for rank, scored in enumerate(results):
            chunk_id = scored.chunk.id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank + 1)
            if chunk_id not in chunks:
                chunks[chunk_id] = scored

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredChunk(chunk=chunks[chunk_id].chunk, score=score, origin="hybrid") for chunk_id, score in ordered
    ]
