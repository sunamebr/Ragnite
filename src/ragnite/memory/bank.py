"""MemoryBank — typed storage and hybrid recall for memory records.

Records live in any ``VectorStore`` as chunks (kind/status/authority in
metadata), embedded on the record text with the subject as contextual prefix,
plus a BM25 index for lexical recall. Recall returns raw ``Evidence`` with
per-record retrieval signals — scoring happens in the ConfidenceScorer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragnite.embed.base import EmbeddingProvider
from ragnite.memory.types import Evidence, MemoryKind, MemoryRecord
from ragnite.retrieve.bm25 import BM25Index
from ragnite.store.base import VectorStore
from ragnite.store.native import NativeVectorStore
from ragnite.types import Chunk

_RESERVED_META = {
    "kind",
    "subject",
    "tags",
    "authority",
    "status",
    "supersedes",
    "created_at",
    "updated_at",
}


def record_to_chunk(record: MemoryRecord) -> Chunk:
    meta: dict[str, Any] = {
        "kind": record.kind.value,
        "subject": record.subject or "",
        "tags": record.tags,
        "authority": record.authority,
        "status": record.status,
        "supersedes": record.supersedes or "",
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    meta.update({k: v for k, v in record.metadata.items() if k not in _RESERVED_META})
    return Chunk(
        id=record.id,
        doc_id=f"memory_{record.kind.value}",
        text=record.text,
        context=record.subject,
        source=record.source,
        metadata=meta,
    )


def record_from_chunk(chunk: Chunk) -> MemoryRecord:
    meta = chunk.metadata
    return MemoryRecord(
        id=chunk.id,
        kind=MemoryKind(meta.get("kind", "fact")),
        text=chunk.text,
        subject=meta.get("subject") or None,
        tags=list(meta.get("tags", [])),
        source=chunk.source,
        authority=float(meta.get("authority", 0.7)),
        status=meta.get("status", "active"),
        supersedes=meta.get("supersedes") or None,
        created_at=float(meta.get("created_at", 0.0)),
        updated_at=float(meta.get("updated_at", 0.0)),
        metadata={k: v for k, v in meta.items() if k not in _RESERVED_META},
    )


def _squash_bm25(score: float) -> float:
    return score / (score + 4.0)


class MemoryBank:
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

    # -- writes -----------------------------------------------------------------

    async def add(self, records: list[MemoryRecord]) -> list[str]:
        if not records:
            return []
        chunks = [record_to_chunk(r) for r in records]
        embeddings = None
        if self.embedder is not None:
            embeddings = await self.embedder.embed_batched([c.index_text for c in chunks])
        await self.store.upsert(chunks, embeddings)
        self._dirty = True
        return [r.id for r in records]

    async def update(self, record: MemoryRecord) -> None:
        await self.add([record])

    async def get(self, record_id: str) -> MemoryRecord | None:
        for chunk in await self.store.all_chunks():
            if chunk.id == record_id:
                return record_from_chunk(chunk)
        return None

    async def supersede(self, old_id: str, replacement: MemoryRecord) -> None:
        """Mark ``old_id`` superseded and store the replacement linked to it."""
        replacement.supersedes = old_id
        old = await self.get(old_id)
        if old is not None:
            old.status = "superseded"
            await self.update(old)
        await self.add([replacement])

    async def delete(self, record_ids: list[str]) -> int:
        removed = await self.store.delete(record_ids)
        if removed:
            self._dirty = True
        return removed

    # -- reads ------------------------------------------------------------------

    async def list(self, kind: MemoryKind | None = None, status: str | None = "active") -> list[MemoryRecord]:
        records = []
        for chunk in await self.store.all_chunks():
            if kind is not None and chunk.metadata.get("kind") != kind.value:
                continue
            if status is not None and chunk.metadata.get("status") != status:
                continue
            records.append(record_from_chunk(chunk))
        return records

    async def count(self) -> int:
        return await self.store.count()

    async def recall(
        self,
        query: str,
        kinds: list[MemoryKind] | None = None,
        k: int = 12,
    ) -> list[Evidence]:
        filters: dict[str, Any] = {"status": "active"}
        if kinds:
            filters["kind"] = [kind.value for kind in kinds]

        merged: dict[str, Evidence] = {}
        if self.embedder is not None:
            query_vector = await self.embedder.embed_query(query)
            for scored in await self.store.search(query_vector, k=k, filters=filters):
                merged[scored.chunk.id] = Evidence(
                    record=record_from_chunk(scored.chunk),
                    similarity=max(0.0, min(1.0, scored.score)),
                    in_dense=True,
                )

        if self._dirty:
            self._bm25.build(await self.store.all_chunks())
            self._dirty = False
        for scored in self._bm25.search(query, k=k, filters=filters):
            squashed = _squash_bm25(scored.score)
            if scored.chunk.id in merged:
                evidence = merged[scored.chunk.id]
                evidence.in_bm25 = True
                evidence.similarity = max(evidence.similarity, squashed)
            else:
                merged[scored.chunk.id] = Evidence(
                    record=record_from_chunk(scored.chunk),
                    similarity=squashed,
                    in_bm25=True,
                )

        ranked = sorted(merged.values(), key=lambda e: e.similarity, reverse=True)[:k]
        for rank, evidence in enumerate(ranked):
            evidence.rank = rank
        return ranked
