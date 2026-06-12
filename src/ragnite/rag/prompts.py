"""Prompt templates used by the engine, contextual enricher, and evaluators."""

from __future__ import annotations

from ragnite.types import ScoredChunk

ANSWER_SYSTEM = """\
You are a retrieval-augmented assistant. Answer the user's question using ONLY \
the numbered sources provided. Rules:
- Cite every factual claim with the matching source marker, e.g. [1] or [2][3].
- If the sources do not contain the answer, say so plainly — do not invent facts.
- Be direct and concise. Answer in the language of the question."""


def format_sources(results: list[ScoredChunk], max_chars_per_chunk: int = 2000) -> str:
    lines: list[str] = []
    for i, scored in enumerate(results, start=1):
        chunk = scored.chunk
        origin = f" — {chunk.source}" if chunk.source else ""
        lines.append(f"[{i}]{origin}\n{chunk.index_text[:max_chars_per_chunk]}")
    return "\n\n".join(lines)


def answer_prompt(query: str, results: list[ScoredChunk]) -> str:
    return f"Sources:\n\n{format_sources(results)}\n\nQuestion: {query}"


CONTEXTUAL_CHUNK_PROMPT = """\
<chunk>
{chunk}
</chunk>
Write a short context (1-2 sentences) situating this chunk within the overall \
document, to improve search retrieval of the chunk. Answer ONLY with the context, \
in the document's language."""


MULTI_QUERY_PROMPT = """\
Generate {n} alternative search queries for the question below. Vary wording, \
specificity, and likely keywords. One query per line, no numbering, no extra text.

Question: {query}"""


FAITHFULNESS_PROMPT = """\
You are grading a RAG answer for faithfulness (groundedness).

Sources:
{context}

Answer to grade:
{answer}

Score from 0.0 to 1.0: the fraction of factual claims in the answer that are \
directly supported by the sources. 1.0 = fully grounded, 0.0 = fabricated."""


RELEVANCY_PROMPT = """\
You are grading how well an answer addresses a question.

Question:
{query}

Answer:
{answer}

Score from 0.0 to 1.0: 1.0 = fully and directly answers the question, \
0.0 = off-topic or non-answer."""


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Score between 0.0 and 1.0."},
        "reasoning": {"type": "string", "description": "One short sentence of justification."},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}
