"""Voyage AI embeddings — the recommended provider for Claude-based stacks."""

from __future__ import annotations

import os

import httpx

from ragnite.embed.base import EmbeddingProvider
from ragnite.errors import ConfigError, RetrievalError

_API_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbedder(EmbeddingProvider):
    name = "voyage"
    batch_size = 128

    def __init__(
        self,
        model: str = "voyage-3.5",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ConfigError("VOYAGE_API_KEY is not set")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _request(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = await self._client.post(
            _API_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self.model, "input_type": input_type},
        )
        if response.status_code != 200:
            raise RetrievalError(f"voyage embeddings failed ({response.status_code}): {response.text[:300]}")
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if vectors and self.dim is None:
            self.dim = len(vectors[0])
        return vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._request(texts, "document")

    async def embed_query(self, query: str) -> list[float]:
        return (await self._request([query], "query"))[0]
