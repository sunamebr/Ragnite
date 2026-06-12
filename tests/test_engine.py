import pytest

from ragnite.errors import ConfigError
from ragnite.rag.engine import RagEngine
from ragnite.store.native import NativeVectorStore


async def test_search_hybrid_finds_relevant_chunk(engine):
    results = await engine.search("Why does Mars look red?", top_k=3)
    assert results
    assert results[0].chunk.doc_id == "doc_mars"


async def test_search_with_filters(engine):
    results = await engine.search("language", top_k=5, filters={"topic": "code"})
    assert results
    assert all(r.chunk.metadata["topic"] == "code" for r in results)


async def test_bm25_only_mode(tmp_path, fake_chat):
    from conftest import sample_docs

    rag = RagEngine(store=NativeVectorStore(tmp_path / "kw"), embedder=None, llm=fake_chat)
    await rag.ingest_documents(sample_docs())
    results = await rag.search("Olympus Mons volcano", top_k=2)
    assert results
    assert results[0].chunk.doc_id == "doc_mars"


async def test_ask_returns_answer_with_citations(engine, fake_chat):
    answer = await engine.ask("Why is Mars red?")
    assert "iron oxide" in answer.text
    assert answer.citations
    assert answer.citations[0].marker == 1
    assert answer.citations[0].doc_id == "doc_mars"
    assert answer.chunks
    # the grounded prompt must contain numbered sources
    prompt = fake_chat.calls[-1]["messages"][-1]["content"]
    assert "[1]" in prompt and "Question:" in prompt


async def test_ask_stream_emits_deltas_then_answer(engine):
    events = [event async for event in engine.ask_stream("Why is Mars red?")]
    assert events[-1].type == "answer"
    deltas = "".join(e.text for e in events if e.type == "delta")
    assert deltas == events[-1].answer.text
    assert events[-1].answer.citations


async def test_ask_without_llm_raises(tmp_path):
    rag = RagEngine(store=NativeVectorStore(tmp_path / "x"))
    await rag.ingest_text("some text", source="t")
    with pytest.raises(ConfigError):
        await rag.ask("anything")


async def test_answer_cache_skips_the_llm_entirely(tmp_path, fake_chat):
    from conftest import sample_docs
    from ragnite.embed.fake import FakeEmbedder
    from ragnite.memory.semcache import AnswerCache

    rag = RagEngine(
        store=NativeVectorStore(tmp_path / "col"),
        embedder=FakeEmbedder(),
        llm=fake_chat,
        answer_cache=AnswerCache(embedder=FakeEmbedder(), path=tmp_path / "ac", threshold=0.8),
    )
    await rag.ingest_documents(sample_docs())

    first = await rag.ask("Why is Mars red?")
    assert first.cached is False
    assert len(fake_chat.calls) == 1

    second = await rag.ask("Why is Mars red?")
    assert second.cached is True
    assert second.text == first.text
    assert len(fake_chat.calls) == 1  # zero LLM calls on the cache hit

    # corpus changed -> cached final answers are invalidated
    await rag.ingest_text("Mars also has dust storms.", source="extra")
    third = await rag.ask("Why is Mars red?")
    assert third.cached is False
    assert len(fake_chat.calls) == 2

    # streamed path serves the cached answer too
    events = [event async for event in rag.ask_stream("Why is Mars red?")]
    assert events[-1].answer.cached is True
    assert len(fake_chat.calls) == 2


async def test_ingest_and_stats(engine):
    stats = await engine.stats()
    assert stats["chunks"] >= 3
    assert stats["bm25"] is True
    await engine.clear()
    assert (await engine.stats())["chunks"] == 0
