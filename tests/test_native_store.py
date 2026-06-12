from conftest import sample_docs
from ragnite.embed.fake import FakeEmbedder
from ragnite.ingest.chunkers import RecursiveChunker
from ragnite.store.native import NativeVectorStore


async def _populate(store: NativeVectorStore):
    embedder = FakeEmbedder()
    chunker = RecursiveChunker()
    chunks = [chunk for doc in sample_docs() for chunk in chunker.chunk(doc)]
    embeddings = await embedder.embed([c.index_text for c in chunks])
    await store.upsert(chunks, embeddings)
    return embedder, chunks


async def test_search_returns_most_similar(tmp_path):
    store = NativeVectorStore(tmp_path / "s")
    embedder, _ = await _populate(store)
    query = await embedder.embed_query("iron oxide dust on mars surface")
    results = await store.search(query, k=2)
    assert results[0].chunk.doc_id == "doc_mars"
    assert results[0].score >= results[1].score


async def test_metadata_filters(tmp_path):
    store = NativeVectorStore(tmp_path / "s")
    embedder, _ = await _populate(store)
    query = await embedder.embed_query("language")
    results = await store.search(query, k=5, filters={"topic": "space"})
    assert results
    assert all(r.chunk.metadata["topic"] == "space" for r in results)
    results_in = await store.search(query, k=5, filters={"topic": ["space", "code"]})
    assert len(results_in) >= len(results)


async def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "persist"
    store = NativeVectorStore(path)
    embedder, chunks = await _populate(store)

    reloaded = NativeVectorStore(path)
    assert await reloaded.count() == len(chunks)
    query = await embedder.embed_query("largest planet storm")
    results = await reloaded.search(query, k=1)
    assert results[0].chunk.doc_id == "doc_jupiter"


async def test_delete_and_upsert_overwrite(tmp_path):
    store = NativeVectorStore(tmp_path / "s")
    _, chunks = await _populate(store)
    before = await store.count()
    removed = await store.delete([chunks[0].id])
    assert removed == 1
    assert await store.count() == before - 1

    # upserting the same chunk id twice must not duplicate
    embedder = FakeEmbedder()
    vec = await embedder.embed([chunks[1].index_text])
    await store.upsert([chunks[1]], vec)
    assert await store.count() == before - 1
