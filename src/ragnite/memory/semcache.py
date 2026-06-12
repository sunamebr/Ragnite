"""Semantic caches keyed by query embedding.

Two caches, two distinct promises — do not conflate them:

- ``SemanticCache`` (verdict cache): stores the recall *verdict* — packed
  context, mode, confidence, signals. A hit skips retrieval and scoring
  entirely and reuses the packed context. The host still spends LLM tokens
  if it forwards that context to a model.
- ``AnswerCache`` (final-answer cache): stores a *finished, generated*
  ``Answer`` (text + citations) for document RAG. A hit returns the final
  answer — this is the only path that is genuinely zero LLM tokens.

Both are keyed by the query embedding (similarity threshold), TTL-bounded,
fall back to normalized exact-match without an embedder, and must be cleared
by callers on writes (``MemoryEngine`` and ``RagEngine`` do this).
"""

from __future__ import annotations

import time
from pathlib import Path

from ragnite.embed.base import EmbeddingProvider
from ragnite.memory.types import MemoryAnswer
from ragnite.store.base import VectorStore
from ragnite.store.native import NativeVectorStore
from ragnite.types import Answer, Chunk, new_id


def _normalize(query: str) -> str:
    return " ".join(query.lower().split())


class _QueryKeyedCache:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        path: str | Path | None = None,
        threshold: float = 0.90,
        ttl_days: float = 7.0,
    ) -> None:
        self.embedder = embedder
        self.store = store or NativeVectorStore(path)
        self.threshold = threshold
        self.ttl_days = ttl_days

    def _fresh(self, created_at: float) -> bool:
        return (time.time() - created_at) <= self.ttl_days * 86400.0

    async def _lookup(self, query: str) -> Chunk | None:
        if self.embedder is not None:
            vector = await self.embedder.embed_query(query)
            hits = await self.store.search(vector, k=1)
            if hits and hits[0].score >= self.threshold:
                chunk = hits[0].chunk
                if self._fresh(float(chunk.metadata.get("created_at", 0.0))):
                    return chunk
            return None
        normalized = _normalize(query)
        for chunk in await self.store.all_chunks():
            if chunk.metadata.get("query_norm") == normalized and self._fresh(
                float(chunk.metadata.get("created_at", 0.0))
            ):
                return chunk
        return None

    async def _store_entry(self, query: str, text: str, payload: str) -> None:
        chunk = Chunk(
            id=new_id("sc"),
            doc_id="semcache",
            text=text,
            metadata={
                "query": query,
                "query_norm": _normalize(query),
                "created_at": time.time(),
                "payload": payload,
            },
        )
        embeddings = None
        if self.embedder is not None:
            embeddings = [await self.embedder.embed_query(query)]
        await self.store.upsert([chunk], embeddings)

    async def clear(self) -> None:
        await self.store.clear()

    async def count(self) -> int:
        return await self.store.count()


class SemanticCache(_QueryKeyedCache):
    """Verdict cache for ``MemoryEngine.recall``.

    Saves: retrieval, scoring, packing. Does NOT save generation tokens by
    itself — the cached payload is the packed context + verdict, not a final
    LLM answer. Pair with ``AnswerCache`` when the final answer should be
    reusable too.
    """

    async def get(self, query: str) -> MemoryAnswer | None:
        chunk = await self._lookup(query)
        if chunk is None:
            return None
        answer = MemoryAnswer.model_validate_json(chunk.metadata["payload"])
        answer.query = query
        answer.cached = True
        return answer

    async def put(self, query: str, answer: MemoryAnswer) -> None:
        await self._store_entry(query, answer.context, answer.model_dump_json(exclude={"evidence"}))


class AnswerCache(_QueryKeyedCache):
    """Final-answer cache for ``RagEngine.ask``.

    A hit returns the previously *generated* ``Answer`` (text + citations):
    zero LLM tokens spent. Higher threshold and shorter TTL than the verdict
    cache by default — serving a wrong final answer is worse than re-running
    recall. Opt in via ``RAGNITE_ANSWER_CACHE=1`` or ``answer_cache=`` on
    ``RagEngine``.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        path: str | Path | None = None,
        threshold: float = 0.93,
        ttl_days: float = 3.0,
    ) -> None:
        super().__init__(embedder=embedder, store=store, path=path, threshold=threshold, ttl_days=ttl_days)

    async def get(self, query: str) -> Answer | None:
        chunk = await self._lookup(query)
        if chunk is None:
            return None
        answer = Answer.model_validate_json(chunk.metadata["payload"])
        answer.cached = True
        return answer

    async def put(self, query: str, answer: Answer) -> None:
        await self._store_entry(query, answer.text, answer.model_dump_json(exclude={"chunks"}))
