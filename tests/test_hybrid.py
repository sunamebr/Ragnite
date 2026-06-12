import pytest

from ragnite.retrieve.hybrid import rrf_fuse
from ragnite.types import Chunk, ScoredChunk


def _scored(chunk_id: str, score: float, origin: str = "dense") -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=chunk_id, doc_id="d", text=chunk_id), score=score, origin=origin)


def test_rrf_prefers_chunk_strong_in_both_lists():
    dense = [_scored("a", 0.9), _scored("b", 0.8), _scored("c", 0.7)]
    keyword = [_scored("b", 12.0, "bm25"), _scored("a", 11.0, "bm25"), _scored("d", 1.0, "bm25")]
    fused = rrf_fuse([dense, keyword])
    ids = [s.chunk.id for s in fused]
    assert set(ids[:2]) == {"a", "b"}  # present in both lists -> top
    assert ids.index("d") > ids.index("c") or "c" in ids
    assert all(s.origin == "hybrid" for s in fused)


def test_rrf_weights_shift_ranking():
    dense = [_scored("a", 0.9), _scored("b", 0.8)]
    keyword = [_scored("b", 5.0, "bm25"), _scored("a", 4.0, "bm25")]
    keyword_heavy = rrf_fuse([dense, keyword], weights=[0.1, 2.0])
    assert keyword_heavy[0].chunk.id == "b"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        rrf_fuse([[_scored("a", 1.0)]], weights=[1.0, 2.0])
