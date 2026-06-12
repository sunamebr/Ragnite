"""Local embeddings via sentence-transformers (optional extra ``ragnite[local]``)."""

from __future__ import annotations

import asyncio

from ragnite.embed.base import EmbeddingProvider
from ragnite.errors import MissingDependencyError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class LocalEmbedder(EmbeddingProvider):
    name = "local"
    batch_size = 64

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise MissingDependencyError("sentence-transformers", "local")
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._lock = asyncio.Lock()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._lock:  # the underlying model is not thread-safe
            vectors = await asyncio.to_thread(
                self._model.encode, texts, normalize_embeddings=True, show_progress_bar=False
            )
        return [vector.tolist() for vector in vectors]
