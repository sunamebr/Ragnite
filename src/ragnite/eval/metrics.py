"""Evaluation metrics: classic retrieval IR metrics + LLM-judge generation scores."""

from __future__ import annotations

import json
import math

from ragnite.llm.base import ChatModel
from ragnite.rag.prompts import FAITHFULNESS_PROMPT, JUDGE_SCHEMA, RELEVANCY_PROMPT
from ragnite.types import ScoredChunk


def _relevant(scored: ScoredChunk, relevant_ids: set[str]) -> bool:
    return scored.chunk.id in relevant_ids or scored.chunk.doc_id in relevant_ids


def hit_at_k(results: list[ScoredChunk], relevant_ids: set[str], k: int) -> float:
    return 1.0 if any(_relevant(r, relevant_ids) for r in results[:k]) else 0.0


def mrr_at_k(results: list[ScoredChunk], relevant_ids: set[str], k: int) -> float:
    for rank, scored in enumerate(results[:k], start=1):
        if _relevant(scored, relevant_ids):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[ScoredChunk], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, scored in enumerate(results[:k], start=1)
        if _relevant(scored, relevant_ids)
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


async def _judge(llm: ChatModel, prompt: str) -> float:
    response = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        json_schema=JUDGE_SCHEMA,
    )
    try:
        score = float(json.loads(response.text)["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


async def faithfulness(llm: ChatModel, context: str, answer: str) -> float:
    """Fraction of the answer's claims supported by the retrieved context."""
    return await _judge(llm, FAITHFULNESS_PROMPT.format(context=context, answer=answer))


async def answer_relevancy(llm: ChatModel, query: str, answer: str) -> float:
    """How directly the answer addresses the question."""
    return await _judge(llm, RELEVANCY_PROMPT.format(query=query, answer=answer))
