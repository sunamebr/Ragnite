# Ragnite Architecture

> This file covers the document-RAG pipeline. The memory & conviction layer
> (typed memory, confidence scoring, answer modes, semantic cache, code
> memory) is documented in [memory.md](memory.md).

## Design principles

1. **Hybrid by default.** Dense-only retrieval fails on exact identifiers,
   names, and error strings; BM25-only fails on paraphrase. Ragnite always
   builds a BM25 index alongside the vector store and fuses with RRF —
   rank-based fusion, so no score normalization is needed across backends.
2. **Every component optional and injectable.** `RagEngine` works with just a
   store (BM25-only). Add an embedder for semantic search, an LLM for answers,
   a reranker for precision. All are interfaces with multiple adapters.
3. **Offline-first core.** `pydantic`, `numpy`, `httpx`, `typer` are the only
   hard dependencies. Provider SDKs and the server stack are extras with lazy
   imports (`MissingDependencyError` tells the user the exact extra).
4. **Async end to end.** Embedding calls batch and run concurrently; the
   contextual enricher is semaphore-bounded; CPU-bound local models run in
   `asyncio.to_thread`.

## Data flow

### Ingestion

```
load_path() -> Document -> chunker_for() -> [Chunk]
   -> ContextualEnricher (optional: LLM writes 1-2 sentence context per chunk,
      full document sent as a cacheable system block)
   -> EmbeddingProvider.embed_batched(chunk.index_text)   # context + body
   -> VectorStore.upsert(chunks, embeddings)
   -> BM25 marked dirty (rebuilt lazily on next search)
```

`Chunk.index_text` (context + body) is what gets embedded and BM25-indexed;
`Chunk.text` (the verbatim body) is what gets cited and shown.

### Query

```
query -> embed_query() -> store.search(k_dense)   ┐
query -> bm25.search(k_bm25)                      ├─> rrf_fuse() -> reranker (optional) -> top_k
                                                  ┘
top_k -> numbered sources prompt -> ChatModel -> answer text
answer text -> [n] marker parsing -> structured Citations
```

## Interfaces

| Interface | Contract | Adapters |
|---|---|---|
| `EmbeddingProvider` | `embed(texts)`, `embed_query(q)` | Voyage, OpenAI-compat, local ST, fake, SQLite cache wrapper |
| `VectorStore` | `upsert / search / delete / count / all_chunks` | Native (NumPy + JSONL/NPY persistence), Qdrant |
| `ChatModel` | `complete(messages, system, json_schema)`, `stream(...)` | Anthropic (official SDK), OpenAI-compat |
| `Reranker` | `rerank(query, results, top_n)` | Cohere, Voyage, listwise LLM |

## Notable implementation details

- **Native store** keeps L2-normalized float32 vectors; cosine similarity is a
  single matrix-vector product. The matrix is cached and invalidated on
  upsert/delete. Exact search — no recall loss — practical to ~10^5–10^6
  vectors per node.
- **BM25** is a dependency-free Okapi implementation with a Unicode-aware
  tokenizer (latin + accents, cyrillic, CJK). The index rebuilds lazily from
  `store.all_chunks()` after ingestion, so it never goes stale relative to the
  store.
- **Structured outputs**: rerank rankings and judge scores request a JSON
  schema (`output_config.format` on Claude, `response_format` on
  OpenAI-compat), with fallback to fused order / zero score on parse failure.
- **Embedding cache** keys on `(provider, model, doc|query, sha256(text))` —
  re-ingesting an unchanged corpus performs zero embedding calls.
- **Refusals**: the Anthropic adapter surfaces `stop_reason == "refusal"` as a
  typed `GenerationError` rather than returning empty text.
