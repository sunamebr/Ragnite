"""Pure-Python BM25 (Okapi) keyword index.

No dependencies, rebuilt in memory from the vector store's chunks. Gives
Ragnite lexical recall out of the box — exact identifiers, names, error
strings — which dense embeddings routinely miss.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from ragnite.store.base import Filters, match_filters
from ragnite.types import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9À-ɏЀ-ӿ一-鿿]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter[str] = Counter()
        for chunk in self._chunks:
            tokens = tokenize(chunk.index_text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            df.update(freqs.keys())
        n = len(self._chunks)
        self._avgdl = (sum(self._doc_lens) / n) if n else 0.0
        self._idf = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        terms = [t for t in tokenize(query) if t in self._idf]
        if not terms:
            return []
        scored: list[tuple[float, int]] = []
        for i, freqs in enumerate(self._doc_freqs):
            if filters and not match_filters(self._chunks[i].metadata, filters):
                continue
            score = 0.0
            dl = self._doc_lens[i] or 1
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += self._idf[term] * tf * (self.k1 + 1.0) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ScoredChunk(chunk=self._chunks[i], score=score, origin="bm25") for score, i in scored[:k]]
