from conftest import sample_docs
from ragnite.ingest.chunkers import RecursiveChunker
from ragnite.retrieve.bm25 import BM25Index, tokenize


def _index() -> BM25Index:
    chunker = RecursiveChunker()
    chunks = [chunk for doc in sample_docs() for chunk in chunker.chunk(doc)]
    index = BM25Index()
    index.build(chunks)
    return index


def test_tokenize_handles_accents_and_case():
    assert tokenize("Programação em Python!") == ["programação", "em", "python"]


def test_bm25_ranks_relevant_doc_first():
    results = _index().search("iron oxide red dust", k=3)
    assert results
    assert results[0].chunk.doc_id == "doc_mars"
    assert results[0].origin == "bm25"


def test_bm25_no_match_returns_empty():
    assert _index().search("zzzz qqqq xxxx", k=3) == []


def test_bm25_filters():
    results = _index().search("planet", k=5, filters={"topic": "code"})
    assert all(r.chunk.metadata["topic"] == "code" for r in results)
