from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from ragnite.embed.fake import FakeEmbedder
from ragnite.llm.base import ChatModel, LLMResponse, Message, SystemPrompt
from ragnite.rag.engine import RagEngine
from ragnite.store.native import NativeVectorStore
from ragnite.types import Document, Usage


def sample_docs() -> list[Document]:
    return [
        Document(
            id="doc_mars",
            text=(
                "Mars is the fourth planet from the Sun. Its red color comes from iron oxide "
                "dust covering the surface. Mars has two small moons, Phobos and Deimos, and "
                "the tallest volcano in the solar system, Olympus Mons."
            ),
            source="mars.md",
            metadata={"topic": "space"},
        ),
        Document(
            id="doc_jupiter",
            text=(
                "Jupiter is the largest planet in the solar system. The Great Red Spot is a "
                "giant storm bigger than Earth. Jupiter has dozens of moons, including Europa, "
                "Ganymede and Io."
            ),
            source="jupiter.md",
            metadata={"topic": "space"},
        ),
        Document(
            id="doc_python",
            text=(
                "Python is a programming language created by Guido van Rossum. It emphasizes "
                "readability, ships a rich standard library, and powers data science, web "
                "backends and automation."
            ),
            source="python.md",
            metadata={"topic": "code"},
        ),
    ]


class FakeChat(ChatModel):
    name = "fakechat"
    model = "fake-1"

    def __init__(self, reply: str = "Mars looks red because of iron oxide dust [1].") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def complete(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 16000,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system, "json_schema": json_schema})
        return LLMResponse(text=self.reply, usage=Usage(input_tokens=10, output_tokens=5), model=self.model)

    async def stream(
        self,
        messages: list[Message],
        system: SystemPrompt | None = None,
        max_tokens: int = 64000,
    ) -> AsyncIterator[str]:
        for word in self.reply.split(" "):
            yield word + " "


@pytest.fixture
def fake_chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
async def engine(tmp_path, fake_chat) -> RagEngine:
    rag = RagEngine(
        store=NativeVectorStore(tmp_path / "collection"),
        embedder=FakeEmbedder(),
        llm=fake_chat,
    )
    await rag.ingest_documents(sample_docs())
    return rag
