"""OpenAI-compatible ``/v1/chat/completions`` provider.

Covers OpenAI, Ollama, vLLM, Groq, LM Studio and other compatible servers.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ragnite.errors import GenerationError
from ragnite.llm.base import ChatModel, LLMResponse, Message, SystemPrompt
from ragnite.types import Usage


def _system_text(system: SystemPrompt | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n\n".join(block.get("text", "") for block in system)


class OpenAICompatChat(ChatModel):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _payload(
        self,
        messages: list[Message],
        system: SystemPrompt | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chat_messages: list[dict[str, str]] = []
        system_text = _system_text(system)
        if system_text:
            chat_messages.append({"role": "system", "content": system_text})
        chat_messages.extend(messages)
        payload: dict[str, Any] = {"model": self.model, "messages": chat_messages, "max_tokens": max_tokens}
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        return payload

    async def complete(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 16000,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, system, max_tokens, json_schema),
        )
        if response.status_code != 200:
            raise GenerationError(f"chat completion failed ({response.status_code}): {response.text[:300]}")
        body = response.json()
        usage = body.get("usage") or {}
        return LLMResponse(
            text=body["choices"][0]["message"].get("content") or "",
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            model=body.get("model"),
        )

    async def stream(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 64000,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, system, max_tokens)
        payload["stream"] = True
        async with self._client.stream(
            "POST", f"{self._base_url}/chat/completions", headers=self._headers(), json=payload
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread())[:300]
                raise GenerationError(f"chat stream failed ({response.status_code}): {detail!r}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
