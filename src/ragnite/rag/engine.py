"""RagEngine — the high-level entry point tying the whole pipeline together."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from pathlib import Path

from ragnite.embed.base import EmbeddingProvider
from ragnite.errors import ConfigError
from ragnite.ingest.chunkers import chunker_for
from ragnite.ingest.loaders import load_path
from ragnite.llm.base import ChatModel, Message
from ragnite.rag.contextual import ContextualEnricher
from ragnite.rag.prompts import ANSWER_SYSTEM, answer_prompt
from ragnite.retrieve.bm25 import BM25Index
from ragnite.retrieve.rerank import Reranker
from ragnite.retrieve.retriever import RetrievalConfig, Retriever
from ragnite.store.base import Filters, VectorStore
from ragnite.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    IngestStats,
    ScoredChunk,
    StreamEvent,
)

_MARKER = re.compile(r"\[(\d+)\]")


class RagEngine:
    """Ingest documents, retrieve with hybrid search, answer with citations.

    Every component is optional except the store:
    - no ``embedder``  -> keyword-only (BM25) retrieval
    - no ``llm``       -> retrieval works, ``ask()`` raises
    - no ``reranker``  -> fused order is used directly
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider | None = None,
        llm: ChatModel | None = None,
        reranker: Reranker | None = None,
        retrieval: RetrievalConfig | None = None,
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
        contextual: bool = False,
        bm25: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.contextual = contextual
        self._bm25 = BM25Index() if bm25 else None
        self._bm25_dirty = True
        self._retriever = Retriever(store, embedder, self._bm25, reranker, retrieval)
        self._enricher = ContextualEnricher(llm) if (contextual and llm) else None

    # -- ingestion -------------------------------------------------------------

    async def ingest_documents(self, docs: list[Document]) -> IngestStats:
        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = chunker_for(doc, self.chunk_size, self.chunk_overlap).chunk(doc)
            if self._enricher:
                await self._enricher.enrich(doc, chunks)
            all_chunks.extend(chunks)
        if not all_chunks:
            return IngestStats(documents=len(docs))

        embeddings = None
        if self.embedder is not None:
            embeddings = await self.embedder.embed_batched([c.index_text for c in all_chunks])
        await self.store.upsert(all_chunks, embeddings)
        self._bm25_dirty = True
        return IngestStats(
            documents=len(docs),
            chunks=len(all_chunks),
            embedded=embeddings is not None,
            contextualized=self._enricher is not None,
        )

    async def ingest_text(
        self, text: str, source: str = "inline", metadata: dict | None = None
    ) -> IngestStats:
        return await self.ingest_documents([Document(text=text, source=source, metadata=metadata or {})])

    async def ingest_path(self, path: str | Path, recursive: bool = True) -> IngestStats:
        return await self.ingest_documents(load_path(path, recursive=recursive))

    async def _refresh_bm25(self) -> None:
        if self._bm25 is not None and self._bm25_dirty:
            self._bm25.build(await self.store.all_chunks())
            self._bm25_dirty = False

    # -- retrieval ---------------------------------------------------------------

    async def search(
        self, query: str, top_k: int | None = None, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        await self._refresh_bm25()
        return await self._retriever.retrieve(query, top_k=top_k, filters=filters)

    # -- generation ---------------------------------------------------------------

    def _require_llm(self) -> ChatModel:
        if self.llm is None:
            raise ConfigError(
                "no LLM configured — set ANTHROPIC_API_KEY (or OPENAI_API_KEY) or pass llm= to RagEngine"
            )
        return self.llm

    @staticmethod
    def _citations(text: str, results: list[ScoredChunk]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[int] = set()
        for match in _MARKER.finditer(text):
            marker = int(match.group(1))
            if marker in seen or not (1 <= marker <= len(results)):
                continue
            seen.add(marker)
            chunk = results[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    snippet=chunk.text[:200],
                )
            )
        return citations

    def _messages(
        self, query: str, results: list[ScoredChunk], history: list[Message] | None
    ) -> list[Message]:
        messages: list[Message] = list(history or [])
        messages.append({"role": "user", "content": answer_prompt(query, results)})
        return messages

    async def ask(
        self,
        query: str,
        top_k: int | None = None,
        filters: Filters | None = None,
        history: list[Message] | None = None,
    ) -> Answer:
        llm = self._require_llm()
        results = await self.search(query, top_k=top_k, filters=filters)
        if not results:
            return Answer(text="No indexed content matched this question.", chunks=[])
        response = await llm.complete(self._messages(query, results, history), system=ANSWER_SYSTEM)
        return Answer(
            text=response.text,
            citations=self._citations(response.text, results),
            chunks=results,
            usage=response.usage,
            model=response.model,
        )

    async def ask_stream(
        self,
        query: str,
        top_k: int | None = None,
        filters: Filters | None = None,
        history: list[Message] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        llm = self._require_llm()
        results = await self.search(query, top_k=top_k, filters=filters)
        if not results:
            yield StreamEvent(type="answer", answer=Answer(text="No indexed content matched this question."))
            return
        parts: list[str] = []
        async for delta in llm.stream(self._messages(query, results, history), system=ANSWER_SYSTEM):
            parts.append(delta)
            yield StreamEvent(type="delta", text=delta)
        text = "".join(parts)
        yield StreamEvent(
            type="answer",
            answer=Answer(text=text, citations=self._citations(text, results), chunks=results),
        )

    # -- maintenance ---------------------------------------------------------------

    async def stats(self) -> dict:
        return {
            "chunks": await self.store.count(),
            "store": type(self.store).__name__,
            "embedder": self.embedder.name if self.embedder else None,
            "llm": getattr(self.llm, "model", None) if self.llm else None,
            "bm25": self._bm25 is not None,
            "contextual": self._enricher is not None,
        }

    async def clear(self) -> None:
        await self.store.clear()
        self._bm25_dirty = True
