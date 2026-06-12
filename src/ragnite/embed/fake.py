"""Deterministic offline embedder.

Hashed bag-of-words projection: cosine similarity correlates with token
overlap. No network, no model download — meant for tests, CI, and demos.
Not a substitute for a real embedding model.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata

from ragnite.embed.base import EmbeddingProvider

_TOKEN = re.compile(r"[a-z0-9À-ɏЀ-ӿ一-鿿]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


class FakeEmbedder(EmbeddingProvider):
    name = "fake"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0
            norm = 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
