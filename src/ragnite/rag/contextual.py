"""Contextual retrieval (Anthropic technique).

Before indexing, an LLM writes a 1-2 sentence context situating each chunk in
its source document. The context is prepended for both embedding and BM25,
which substantially cuts retrieval failures on ambiguous chunks.

The full document is sent as a cacheable system block, so with Claude the
per-chunk cost is dominated by cache reads, not re-processing the document.
"""

from __future__ import annotations

import asyncio
import logging

from ragnite.llm.base import ChatModel
from ragnite.rag.prompts import CONTEXTUAL_CHUNK_PROMPT
from ragnite.types import Chunk, Document

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 80_000


class ContextualEnricher:
    def __init__(self, llm: ChatModel, concurrency: int = 8) -> None:
        self._llm = llm
        self._semaphore = asyncio.Semaphore(concurrency)

    async def enrich(self, doc: Document, chunks: list[Chunk]) -> None:
        """Fill ``chunk.context`` in place. Failures degrade to no context."""
        if len(chunks) < 2:
            return  # a single chunk needs no situating
        system = [
            {
                "type": "text",
                "text": f"<document>\n{doc.text[:_MAX_DOC_CHARS]}\n</document>",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        await asyncio.gather(*(self._one(system, chunk) for chunk in chunks))

    async def _one(self, system: list[dict], chunk: Chunk) -> None:
        async with self._semaphore:
            try:
                response = await self._llm.complete(
                    messages=[{"role": "user", "content": CONTEXTUAL_CHUNK_PROMPT.format(chunk=chunk.text)}],
                    system=system,
                    max_tokens=300,
                )
                context = response.text.strip()
                if context:
                    chunk.context = context
            except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                logger.warning("contextual enrichment failed for %s: %s", chunk.id, exc)
