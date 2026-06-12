"""Claude via the official Anthropic SDK (optional extra ``ragnite[anthropic]``).

Uses adaptive thinking and structured outputs (``output_config.format``).
Sampling parameters are intentionally not exposed — current Opus-tier models
reject them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ragnite.errors import GenerationError, MissingDependencyError
from ragnite.llm.base import ChatModel, LLMResponse, Message, SystemPrompt
from ragnite.types import Usage

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover - optional dependency
    _anthropic = None

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicChat(ChatModel):
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        thinking: bool = True,
    ) -> None:
        if _anthropic is None:
            raise MissingDependencyError("anthropic", "anthropic")
        self.model = model
        self._thinking = thinking
        self._client = _anthropic.AsyncAnthropic(api_key=api_key) if api_key else _anthropic.AsyncAnthropic()

    def _kwargs(
        self,
        messages: list[Message],
        system: SystemPrompt | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        if self._thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if json_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        return kwargs

    async def complete(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 16000,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        try:
            response = await self._client.messages.create(
                **self._kwargs(messages, system, max_tokens, json_schema)
            )
        except _anthropic.APIError as exc:
            raise GenerationError(f"anthropic request failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise GenerationError("the model declined this request (stop_reason=refusal)")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            model=response.model,
        )

    async def stream(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 64000,
    ) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(**self._kwargs(messages, system, max_tokens)) as stream:
                async for text in stream.text_stream:
                    yield text
        except _anthropic.APIError as exc:
            raise GenerationError(f"anthropic stream failed: {exc}") from exc
