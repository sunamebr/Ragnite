"""Chat model interface."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from ragnite.types import Usage

# Messages follow the common {"role": "user"|"assistant", "content": str} shape.
Message = dict[str, str]
# System prompts may be a plain string or provider-native content blocks
# (e.g. Anthropic text blocks carrying cache_control).
SystemPrompt = str | list[dict[str, Any]]


class LLMResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None


class ChatModel(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 16000,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One-shot completion. When ``json_schema`` is given the response text
        is guaranteed (or best-effort, per provider) to be JSON matching it."""

    @abc.abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 64000,
    ) -> AsyncIterator[str]:
        """Stream text deltas."""
