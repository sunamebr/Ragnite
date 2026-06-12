"""Retrieval orchestration: dense + BM25 -> RRF -> optional rerank."""

from __future__ import annotations

from pydantic import BaseModel

from ragnite.embed.base import EmbeddingProvider
from ragnite.retrieve.bm25 import BM25Index
from ragnite.retrieve.hybrid import rrf_fuse
from ragnite.retrieve.rerank import Reranker
from ragnite.store.base import Filters, VectorStore
from ragnite.types import ScoredChunk


class RetrievalConfig(BaseModel):
    top_k: int = 6
    k_dense: int = 24
    k_bm25: int = 24
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rerank_candidates: int = 24


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Filters | None = None,
    ) -> list[ScoredChunk]:
        cfg = self.config
        top_k = top_k or cfg.top_k

        lists: list[list[ScoredChunk]] = []
        weights: list[float] = []
        if self.embedder is not None:
            query_vector = await self.embedder.embed_query(query)
            lists.append(await self.store.search(query_vector, k=cfg.k_dense, filters=filters))
            weights.append(cfg.dense_weight)
        if self.bm25 is not None and len(self.bm25):
            lists.append(self.bm25.search(query, k=cfg.k_bm25, filters=filters))
            weights.append(cfg.bm25_weight)

        non_empty = [(lst, w) for lst, w in zip(lists, weights, strict=True) if lst]
        if not non_empty:
            return []
        if len(non_empty) == 1:
            fused = non_empty[0][0]
        else:
            fused = rrf_fuse([lst for lst, _ in non_empty], k=cfg.rrf_k, weights=[w for _, w in non_empty])

        if self.reranker is not None:
            candidates = fused[: cfg.rerank_candidates]
            return (await self.reranker.rerank(query, candidates, top_n=top_k))[:top_k]
        return fused[:top_k]
