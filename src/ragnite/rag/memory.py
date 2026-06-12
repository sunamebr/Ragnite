"""Memory primitives.

``ConversationMemory`` — short-term, in-process turn buffer.
``VectorMemory``       — long-term semantic memory backed by its own vector
collection; this is what the MCP server exposes as ``remember``/``recall`` so
agents (e.g. Claude) get persistent memory across sessions.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from ragnite.embed.base import EmbeddingProvider
from ragnite.llm.base import Message
from ragnite.retrieve.bm25 import BM25Index
from ragnite.retrieve.hybrid import rrf_fuse
from ragnite.store.base import VectorStore
from ragnite.store.native import NativeVectorStore
from ragnite.types import Chunk, ScoredChunk, new_id


class ConversationMemory:
    def __init__(self, max_turns: int = 20) -> None:
        self._turns: deque[Message] = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})

    def messages(self) -> list[Message]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()


class VectorMemory:
    """Persistent semantic memory: store facts, recall by similarity."""

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store or NativeVectorStore(path)
        self._bm25 = BM25Index()
        self._dirty = True

    async def remember(self, fact: str, metadata: dict | None = None) -> str:
        meta = {"kind": "memory", "stored_at": time.time(), **(metadata or {})}
        chunk = Chunk(id=new_id("mem"), doc_id="memory", text=fact.strip(), metadata=meta)
        embeddings = None
        if self.embedder is not None:
            embeddings = [await self.embedder.embed_query(chunk.text)]
        await self.store.upsert([chunk], embeddings)
        self._dirty = True
        return chunk.id

    async def recall(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if self._dirty:
            self._bm25.build(await self.store.all_chunks())
            self._dirty = False
        lists: list[list[ScoredChunk]] = []
        if self.embedder is not None:
            vector = await self.embedder.embed_query(query)
            dense = await self.store.search(vector, k=k * 4)
            if dense:
                lists.append(dense)
        keyword = self._bm25.search(query, k=k * 4)
        if keyword:
            lists.append(keyword)
        if not lists:
            return []
        fused = lists[0] if len(lists) == 1 else rrf_fuse(lists)
        return fused[:k]

    async def forget(self, memory_id: str) -> bool:
        return await self.store.delete([memory_id]) > 0

    async def count(self) -> int:
        return await self.store.count()
