# Changelog

All notable changes to Ragnite are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); versioning follows SemVer.

## [0.2.0] - 2026-06-12

### Changed
- **Repositioned as a Confidence-Aware RAG Memory Engine** — a memory and
  conviction layer for LLMs and coding agents, not just document RAG.

### Added
- `ragnite.memory` subsystem:
  - **Typed memory bank**: Factual, Decision (with supersession chains),
    Episodic, and Code memory over any vector store + BM25.
  - **Confidence Scorer**: similarity, source count, dense↔keyword agreement,
    per-kind recency half-life, source authority, conflict detection —
    relevance-capped so weak matches never score high.
  - **Answer modes**: `direct` / `cautious` / `ask_clarification` /
    `search_more` / `refuse_guess`, each with an instruction for the LLM.
  - **Context Packer**: smallest sufficient context under a token budget,
    near-duplicate suppression, compact provenance headers.
  - **Semantic Cache**: equivalent questions reuse prior verdicts with zero
    retrieval/scoring; TTL-bounded; invalidated on writes.
  - **Code Memory**: incremental repository indexing (AST for Python, symbol
    extraction elsewhere) — files, symbols, imports, endpoints, tests, and an
    import graph.
  - `MemoryEngine.recall()` returning `MemoryAnswer` (mode + confidence +
    signals + packed context + suggestion).
- `build_memory_engine()` factory; `RAGNITE_MEMORY_BUDGET`,
  `RAGNITE_CACHE_THRESHOLD`, `RAGNITE_CACHE_TTL_DAYS` env knobs.
- MCP tools: `recall`, `remember`, `remember_decision`, `index_repo`, `forget`.
- CLI: `ragnite remember`, `ragnite recall`, `ragnite index-code`.
- HTTP API: `/v1/memory/remember`, `/v1/memory/recall`,
  `/v1/memory/index_code`, `/v1/memory/stats`.
- `docs/memory.md` design document.

## [0.1.0] - 2026-06-12

### Added
- Hybrid retrieval: dense vectors + built-in BM25 + Reciprocal Rank Fusion.
- Native zero-dependency vector store (NumPy, persisted to disk) and Qdrant adapter.
- Embedding providers: Voyage AI, OpenAI-compatible, local sentence-transformers,
  deterministic fake; SQLite embedding cache.
- LLM providers: Anthropic Claude (official SDK, adaptive thinking, structured
  outputs) and OpenAI-compatible.
- Contextual retrieval (LLM-written chunk context with prompt caching).
- Rerankers: Cohere, Voyage, and listwise LLM reranking.
- Grounded answers with `[n]` citations, sync and SSE streaming.
- Vector memory (`remember`/`recall`) and conversation memory.
- MCP server (search, ask, ingest, memory) for Claude Code / Desktop / any MCP host.
- FastAPI HTTP service with optional bearer auth.
- Typer CLI: `ingest`, `query`, `ask`, `serve`, `mcp`, `eval`, `stats`, `clear`.
- Evaluation: hit@k, MRR, nDCG + LLM-judge faithfulness/relevancy.
- Loaders: text/markdown/code/html/json out of the box; PDF and DOCX via extras.
- Docker image + docker-compose (API + Qdrant), CI (lint/format/tests on
  Linux + Windows), PyPI release workflow.
