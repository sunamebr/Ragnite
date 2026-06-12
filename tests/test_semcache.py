import asyncio

from ragnite.embed.fake import FakeEmbedder
from ragnite.memory.semcache import AnswerCache, SemanticCache
from ragnite.memory.types import MemoryAnswer
from ragnite.types import Answer, Citation


def _answer(query: str) -> MemoryAnswer:
    return MemoryAnswer(
        query=query,
        mode="direct",
        confidence=0.91,
        suggestion="answer directly",
        context="the database listens on port 5432",
        tokens=9,
    )


async def test_semantic_hit_on_equivalent_query(tmp_path):
    cache = SemanticCache(embedder=FakeEmbedder(), path=tmp_path / "sc", threshold=0.8)
    query = "what port does the database use"
    await cache.put(query, _answer(query))

    exact = await cache.get(query)
    assert exact is not None and exact.cached is True
    assert exact.confidence == 0.91 and exact.context.endswith("5432")

    # same bag of words, different order — semantically equivalent for the cache
    paraphrase = await cache.get("the database use what port")
    assert paraphrase is not None and paraphrase.cached is True

    miss = await cache.get("favorite color of the office cat")
    assert miss is None


async def test_ttl_expiry(tmp_path):
    cache = SemanticCache(embedder=FakeEmbedder(), path=tmp_path / "sc", ttl_days=0.0)
    await cache.put("q", _answer("q"))
    await asyncio.sleep(0.02)
    assert await cache.get("q") is None


async def test_exact_match_fallback_without_embedder(tmp_path):
    cache = SemanticCache(embedder=None, path=tmp_path / "sc")
    await cache.put("Which DB do we use?", _answer("Which DB do we use?"))
    hit = await cache.get("  which db DO we use?  ".replace("  ", " "))
    assert hit is not None and hit.cached is True
    assert await cache.get("a different question") is None


async def test_clear(tmp_path):
    cache = SemanticCache(embedder=FakeEmbedder(), path=tmp_path / "sc")
    await cache.put("q", _answer("q"))
    assert await cache.count() == 1
    await cache.clear()
    assert await cache.count() == 0


async def test_answer_cache_returns_final_answer(tmp_path):
    cache = AnswerCache(embedder=FakeEmbedder(), path=tmp_path / "ac", threshold=0.8)
    final = Answer(
        text="Mars is red because of iron oxide [1].",
        citations=[Citation(marker=1, chunk_id="c1", doc_id="doc_mars", snippet="...")],
    )
    await cache.put("why is mars red", final)

    hit = await cache.get("why is mars red")
    assert hit is not None
    assert hit.cached is True
    assert hit.text == final.text
    assert hit.citations[0].doc_id == "doc_mars"
    assert hit.chunks == []  # heavy chunk payloads are not cached

    assert await cache.get("how do volcanoes form on jupiter moons") is None
