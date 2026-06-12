"""Rerankers — second-stage precision on top of hybrid recall."""

from __future__ import annotations

import abc
import json
import os

import httpx

from ragnite.errors import ConfigError, RetrievalError
from ragnite.llm.base import ChatModel
from ragnite.types import ScoredChunk


class Reranker(abc.ABC):
    @abc.abstractmethod
    async def rerank(self, query: str, results: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


def _as_reranked(results: list[ScoredChunk], order: list[tuple[int, float]]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=results[index].chunk, score=score, origin="rerank")
        for index, score in order
        if 0 <= index < len(results)
    ]


class CohereReranker(Reranker):
    def __init__(self, model: str = "rerank-v3.5", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self._api_key:
            raise ConfigError("COHERE_API_KEY is not set")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, results: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not results:
            return []
        response = await self._client.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [r.chunk.index_text for r in results],
                "top_n": min(top_n, len(results)),
            },
        )
        if response.status_code != 200:
            raise RetrievalError(f"cohere rerank failed ({response.status_code}): {response.text[:300]}")
        order = [(item["index"], float(item["relevance_score"])) for item in response.json()["results"]]
        return _as_reranked(results, order)


class VoyageReranker(Reranker):
    def __init__(self, model: str = "rerank-2", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ConfigError("VOYAGE_API_KEY is not set")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, results: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not results:
            return []
        response = await self._client.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [r.chunk.index_text for r in results],
                "top_k": min(top_n, len(results)),
            },
        )
        if response.status_code != 200:
            raise RetrievalError(f"voyage rerank failed ({response.status_code}): {response.text[:300]}")
        order = [(item["index"], float(item["relevance_score"])) for item in response.json()["data"]]
        return _as_reranked(results, order)


_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Passage numbers ordered from most to least relevant.",
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


class LLMReranker(Reranker):
    """Listwise reranking with any ChatModel. No extra vendor, works everywhere."""

    def __init__(self, llm: ChatModel, max_passage_chars: int = 600) -> None:
        self._llm = llm
        self._max_chars = max_passage_chars

    async def rerank(self, query: str, results: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not results:
            return []
        passages = "\n\n".join(
            f"[{i}] {r.chunk.index_text[: self._max_chars]}" for i, r in enumerate(results)
        )
        prompt = (
            "Rank the passages below by relevance to the query.\n"
            f"Query: {query}\n\nPassages:\n{passages}\n\n"
            "Return the passage numbers ordered from most to least relevant."
        )
        response = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            json_schema=_RANKING_SCHEMA,
        )
        try:
            ranking = json.loads(response.text)["ranking"]
            order = [(int(index), 1.0 / (rank + 1)) for rank, index in enumerate(ranking)]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return results[:top_n]  # fall back to the fused order
        seen: set[int] = set()
        deduped = [(i, s) for i, s in order if not (i in seen or seen.add(i))]
        return _as_reranked(results, deduped)[:top_n]
