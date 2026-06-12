"""SemanticCache — reuse answers for equivalent questions.

Cache entries are keyed by the *query embedding*; a new query that lands above
the similarity threshold returns the previously packed context + verdict with
zero retrieval, zero scoring and zero LLM tokens. Without an embedder it
degrades to normalized exact-match.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ragnite.embed.base import EmbeddingProvider
from ragnite.memory.types import MemoryAnswer
from ragnite.store.base import VectorStore
from ragnite.store.native import NativeVectorStore
from ragnite.types import Chunk, new_id


def _normalize(query: str) -> str:
    return " ".join(query.lower().split())


class SemanticCache:
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

    @staticmethod
    def _to_answer(chunk: Chunk, query: str) -> MemoryAnswer:
        answer = MemoryAnswer.model_validate_json(chunk.metadata["payload"])
        answer.query = query
        answer.cached = True
        return answer

    async def get(self, query: str) -> MemoryAnswer | None:
        if self.embedder is not None:
            vector = await self.embedder.embed_query(query)
            hits = await self.store.search(vector, k=1)
            if hits and hits[0].score >= self.threshold:
                chunk = hits[0].chunk
                if self._fresh(float(chunk.metadata.get("created_at", 0.0))):
                    return self._to_answer(chunk, query)
            return None
        normalized = _normalize(query)
        for chunk in await self.store.all_chunks():
            if chunk.metadata.get("query_norm") == normalized and self._fresh(
                float(chunk.metadata.get("created_at", 0.0))
            ):
                return self._to_answer(chunk, query)
        return None

    async def put(self, query: str, answer: MemoryAnswer) -> None:
        payload = answer.model_dump(exclude={"evidence"})
        chunk = Chunk(
            id=new_id("sc"),
            doc_id="semcache",
            text=answer.context,
            metadata={
                "query": query,
                "query_norm": _normalize(query),
                "created_at": time.time(),
                "payload": json.dumps(payload, ensure_ascii=False),
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
