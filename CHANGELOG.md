# Changelog

All notable changes to Ragnite are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); versioning follows SemVer.

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
