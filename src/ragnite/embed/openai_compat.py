"""OpenAI-compatible ``/v1/embeddings`` provider.

Works with OpenAI, Ollama, vLLM, Jina, LM Studio and any server that speaks
the same wire format — point ``base_url`` at it.
"""

from __future__ import annotations

import os

import httpx

from ragnite.embed.base import EmbeddingProvider
from ragnite.errors import RetrievalError


class OpenAICompatEmbedder(EmbeddingProvider):
    name = "openai"
    batch_size = 256

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            headers=headers,
            json={"input": texts, "model": self.model},
        )
        if response.status_code != 200:
            raise RetrievalError(f"embeddings failed ({response.status_code}): {response.text[:300]}")
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if vectors and self.dim is None:
            self.dim = len(vectors[0])
        return vectors
