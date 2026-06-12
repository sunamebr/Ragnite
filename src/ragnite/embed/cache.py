"""SQLite-backed embedding cache.

Keyed on (provider, model-ish name, text hash). Re-ingesting the same corpus
costs zero embedding calls. Wrap any provider:

    embedder = EmbeddingCache(VoyageEmbedder(), path=".ragnite/embeddings.db")
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import struct
from pathlib import Path

from ragnite.embed.base import EmbeddingProvider


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EmbeddingCache(EmbeddingProvider):
    name = "cache"

    def __init__(self, inner: EmbeddingProvider, path: str | Path) -> None:
        self.inner = inner
        self.name = f"cache({inner.name})"
        self.batch_size = inner.batch_size
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self._db.commit()
        self._lock = asyncio.Lock()

    @property
    def dim(self) -> int | None:  # type: ignore[override]
        return self.inner.dim

    def _key(self, text: str, kind: str) -> str:
        model = getattr(self.inner, "model", self.inner.name)
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"{self.inner.name}:{model}:{kind}:{digest}"

    async def _lookup(self, keys: list[str]) -> dict[str, list[float]]:
        async with self._lock:
            placeholders = ",".join("?" for _ in keys)
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", keys
            ).fetchall()
        return {key: _unpack(blob) for key, blob in rows}

    async def _store(self, items: dict[str, list[float]]) -> None:
        async with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(key, _pack(vector)) for key, vector in items.items()],
            )
            self._db.commit()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text, "doc") for text in texts]
        cached = await self._lookup(keys) if keys else {}
        missing = [i for i, key in enumerate(keys) if key not in cached]
        if missing:
            fresh = await self.inner.embed_batched([texts[i] for i in missing])
            new_items = {keys[i]: vector for i, vector in zip(missing, fresh, strict=True)}
            await self._store(new_items)
            cached.update(new_items)
        return [cached[key] for key in keys]

    async def embed_query(self, query: str) -> list[float]:
        key = self._key(query, "query")
        cached = await self._lookup([key])
        if key in cached:
            return cached[key]
        vector = await self.inner.embed_query(query)
        await self._store({key: vector})
        return vector
