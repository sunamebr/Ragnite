"""Qdrant adapter (optional extra ``ragnite[qdrant]``) for horizontal scale."""

from __future__ import annotations

import uuid

from ragnite.errors import MissingDependencyError
from ragnite.store.base import Filters, VectorStore
from ragnite.types import Chunk, ScoredChunk

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError:  # pragma: no cover - optional dependency
    AsyncQdrantClient = None
    models = None


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "ragnite",
        api_key: str | None = None,
    ) -> None:
        if AsyncQdrantClient is None:
            raise MissingDependencyError("qdrant-client", "qdrant")
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._ready = False

    async def _ensure_collection(self, dim: int) -> None:
        if self._ready:
            return
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        self._ready = True

    @staticmethod
    def _qdrant_filter(filters: Filters | None) -> models.Filter | None:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            field = f"metadata.{key}"
            if isinstance(value, list):
                conditions.append(models.FieldCondition(key=field, match=models.MatchAny(any=value)))
            else:
                conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
        return models.Filter(must=conditions)

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        if not embeddings:
            raise ValueError("QdrantVectorStore requires embeddings; configure an embedding provider")
        await self._ensure_collection(len(embeddings[0]))
        points = [
            models.PointStruct(id=_point_id(chunk.id), vector=vector, payload=chunk.model_dump())
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self, embedding: list[float], k: int = 10, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        if not await self._client.collection_exists(self._collection):
            return []
        response = await self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=k,
            query_filter=self._qdrant_filter(filters),
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(point.payload), score=float(point.score), origin="dense")
            for point in response.points
        ]

    async def delete(self, chunk_ids: list[str]) -> int:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=[_point_id(c) for c in chunk_ids]),
        )
        return len(chunk_ids)

    async def count(self) -> int:
        if not await self._client.collection_exists(self._collection):
            return 0
        return (await self._client.count(self._collection)).count

    async def all_chunks(self) -> list[Chunk]:
        if not await self._client.collection_exists(self._collection):
            return []
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection, limit=512, offset=offset, with_payload=True
            )
            chunks.extend(Chunk.model_validate(point.payload) for point in points)
            if offset is None:
                break
        return chunks
