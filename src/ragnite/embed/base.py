"""Embedding provider interface."""

from __future__ import annotations

import abc


class EmbeddingProvider(abc.ABC):
    """Async embedding backend. Implementations must be safe for concurrent calls."""

    name: str = "base"
    dim: int | None = None
    batch_size: int = 128

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (storage side)."""

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query. Providers with asymmetric models override this."""
        return (await self.embed([query]))[0]

    async def embed_batched(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(await self.embed(texts[start : start + self.batch_size]))
        return out
