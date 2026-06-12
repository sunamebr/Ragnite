"""Built-in vector store: NumPy exact cosine search with disk persistence.

Zero external services. Exact (brute-force) search is fast well into the
hundreds of thousands of vectors; beyond that, switch to the Qdrant adapter.

Layout on disk (one directory per collection):
    chunks.jsonl   — one chunk per line
    vectors.npy    — float32 matrix, row-aligned with chunks.jsonl
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from ragnite.store.base import Filters, VectorStore, match_filters
from ragnite.types import Chunk, ScoredChunk


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class NativeVectorStore(VectorStore):
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._vectors: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []
        self._lock = asyncio.Lock()
        if self._path:
            self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        assert self._path is not None
        chunks_file = self._path / "chunks.jsonl"
        if not chunks_file.exists():
            return
        with chunks_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = Chunk.model_validate_json(line)
                    self._chunks[chunk.id] = chunk
                    self._order.append(chunk.id)
        vectors_file = self._path / "vectors.npy"
        if vectors_file.exists():
            matrix = np.load(vectors_file)
            if matrix.shape[0] == len(self._order):
                for chunk_id, row in zip(self._order, matrix, strict=True):
                    self._vectors[chunk_id] = row

    def _save(self) -> None:
        if not self._path:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        with (self._path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk_id in self._order:
                handle.write(self._chunks[chunk_id].model_dump_json() + "\n")
        if len(self._vectors) == len(self._order) and self._order:
            matrix = np.stack([self._vectors[chunk_id] for chunk_id in self._order])
            np.save(self._path / "vectors.npy", matrix)
        elif self._path.joinpath("vectors.npy").exists() and not self._vectors:
            self._path.joinpath("vectors.npy").unlink()

    # -- VectorStore ----------------------------------------------------------

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        async with self._lock:
            for i, chunk in enumerate(chunks):
                if chunk.id not in self._chunks:
                    self._order.append(chunk.id)
                self._chunks[chunk.id] = chunk
                if embeddings is not None:
                    vector = np.asarray(embeddings[i], dtype=np.float32)
                    norm = np.linalg.norm(vector)
                    self._vectors[chunk.id] = vector / norm if norm else vector
            self._matrix = None
            self._save()

    def _ensure_matrix(self) -> None:
        embedded = [chunk_id for chunk_id in self._order if chunk_id in self._vectors]
        self._matrix_ids = embedded
        self._matrix = np.stack([self._vectors[c] for c in embedded]) if embedded else None

    async def search(
        self, embedding: list[float], k: int = 10, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        async with self._lock:
            if self._matrix is None:
                self._ensure_matrix()
            if self._matrix is None or not len(self._matrix_ids):
                return []
            query = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm:
                query = query / norm
            scores = self._matrix @ query
            order = np.argsort(scores)[::-1]
            results: list[ScoredChunk] = []
            for idx in order:
                chunk = self._chunks[self._matrix_ids[int(idx)]]
                if not match_filters(chunk.metadata, filters):
                    continue
                results.append(ScoredChunk(chunk=chunk, score=float(scores[int(idx)]), origin="dense"))
                if len(results) >= k:
                    break
            return results

    async def delete(self, chunk_ids: list[str]) -> int:
        async with self._lock:
            removed = 0
            for chunk_id in chunk_ids:
                if chunk_id in self._chunks:
                    del self._chunks[chunk_id]
                    self._vectors.pop(chunk_id, None)
                    removed += 1
            if removed:
                self._order = [c for c in self._order if c in self._chunks]
                self._matrix = None
                self._save()
            return removed

    async def count(self) -> int:
        return len(self._chunks)

    async def all_chunks(self) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in self._order]
