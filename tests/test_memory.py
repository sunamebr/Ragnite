from ragnite.embed.fake import FakeEmbedder
from ragnite.rag.memory import ConversationMemory, VectorMemory


async def test_vector_memory_recall(tmp_path):
    memory = VectorMemory(embedder=FakeEmbedder(), path=tmp_path / "mem")
    await memory.remember("The user's favorite color is blue.")
    await memory.remember("The user's dog is named Rex.")
    await memory.remember("Production database runs on port 5432.")

    results = await memory.recall("what color does the user like?", k=1)
    assert results
    assert "blue" in results[0].chunk.text

    assert await memory.count() == 3


async def test_vector_memory_persists(tmp_path):
    path = tmp_path / "mem"
    first = VectorMemory(embedder=FakeEmbedder(), path=path)
    memory_id = await first.remember("Deploy happens every Friday.")

    second = VectorMemory(embedder=FakeEmbedder(), path=path)
    results = await second.recall("when is the deploy?", k=1)
    assert results and "Friday" in results[0].chunk.text
    assert await second.forget(memory_id)
    assert await second.count() == 0


def test_conversation_memory_window():
    convo = ConversationMemory(max_turns=2)
    for i in range(5):
        convo.add("user", f"q{i}")
        convo.add("assistant", f"a{i}")
    messages = convo.messages()
    assert len(messages) == 4
    assert messages[-1]["content"] == "a4"
