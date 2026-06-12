"""Evaluation runner.

Dataset format: JSONL, one case per line:
    {"query": "...", "relevant_ids": ["doc_or_chunk_id", ...], "reference": "optional gold answer"}

Retrieval metrics need only ``relevant_ids``. Generation metrics (``--judge``)
additionally run the engine end-to-end and grade answers with an LLM judge.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ragnite.eval.metrics import answer_relevancy, faithfulness, hit_at_k, mrr_at_k, ndcg_at_k
from ragnite.rag.engine import RagEngine
from ragnite.rag.prompts import format_sources


class EvalCase(BaseModel):
    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    reference: str | None = None


class EvalReport(BaseModel):
    cases: int
    k: int
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    faithfulness: float | None = None
    answer_relevancy: float | None = None


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase.model_validate_json(line))
    return cases


async def run_eval(
    engine: RagEngine,
    cases: list[EvalCase],
    k: int = 6,
    judge: bool = False,
) -> EvalReport:
    if not cases:
        raise ValueError("empty evaluation dataset")
    hits: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    faith_scores: list[float] = []
    rel_scores: list[float] = []

    for case in cases:
        results = await engine.search(case.query, top_k=k)
        relevant = set(case.relevant_ids)
        if relevant:
            hits.append(hit_at_k(results, relevant, k))
            mrrs.append(mrr_at_k(results, relevant, k))
            ndcgs.append(ndcg_at_k(results, relevant, k))
        if judge and engine.llm is not None:
            answer = await engine.ask(case.query, top_k=k)
            context = format_sources(answer.chunks)
            faith_scores.append(await faithfulness(engine.llm, context, answer.text))
            rel_scores.append(await answer_relevancy(engine.llm, case.query, answer.text))

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return EvalReport(
        cases=len(cases),
        k=k,
        hit_rate=avg(hits),
        mrr=avg(mrrs),
        ndcg=avg(ndcgs),
        faithfulness=avg(faith_scores) if faith_scores else None,
        answer_relevancy=avg(rel_scores) if rel_scores else None,
    )
