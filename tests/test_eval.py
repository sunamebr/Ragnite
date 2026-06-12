from ragnite.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k
from ragnite.eval.runner import EvalCase, run_eval
from ragnite.types import Chunk, ScoredChunk


def _results(*doc_ids: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(id=f"c{i}", doc_id=doc_id, text="x"), score=1.0 - i * 0.1)
        for i, doc_id in enumerate(doc_ids)
    ]


def test_retrieval_metrics():
    results = _results("d1", "d2", "d3")
    assert hit_at_k(results, {"d2"}, k=3) == 1.0
    assert hit_at_k(results, {"d9"}, k=3) == 0.0
    assert mrr_at_k(results, {"d2"}, k=3) == 0.5
    assert ndcg_at_k(results, {"d1"}, k=3) == 1.0
    assert 0.0 < ndcg_at_k(results, {"d3"}, k=3) < 1.0


async def test_run_eval_end_to_end(engine):
    cases = [
        EvalCase(query="Why is Mars red?", relevant_ids=["doc_mars"]),
        EvalCase(query="largest planet with a giant storm", relevant_ids=["doc_jupiter"]),
    ]
    report = await run_eval(engine, cases, k=4)
    assert report.cases == 2
    assert report.hit_rate == 1.0
    assert report.mrr > 0.4
    assert report.faithfulness is None  # judge disabled
