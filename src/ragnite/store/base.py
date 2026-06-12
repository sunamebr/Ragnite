"""Vector store interface and metadata filter matching."""

from __future__ import annotations

import abc
from typing import Any

from ragnite.types import Chunk, ScoredChunk

Filters = dict[str, Any]


def match_filters(metadata: dict[str, Any], filters: Filters | None) -> bool:
    """Equality match per key; a list value means "metadata value in list"."""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class VectorStore(abc.ABC):
    """Stores chunks and (optionally) their embeddings."""

    @abc.abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None: ...

    @abc.abstractmethod
    async def search(
        self, embedding: list[float], k: int = 10, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...

    @abc.abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int: ...

    @abc.abstractmethod
    async def count(self) -> int: ...

    @abc.abstractmethod
    async def all_chunks(self) -> list[Chunk]:
        """Full chunk dump — used to build the keyword (BM25) index."""

    async def clear(self) -> None:
        ids = [chunk.id for chunk in await self.all_chunks()]
        if ids:
            await self.delete(ids)
