# Changelog

All notable changes to Ragnite are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); versioning follows SemVer.

## [0.3.1] - 2026-06-12

Release polish — consistency, docs and release hygiene. No new features.

### Fixed
- Unified the test count across CHANGELOG and release notes (103 tests; the
  v0.3.0 entry previously said 99 from before the final regression tests).
- Markdown rendering: raw `<ragnite-context ...>` markup in README/CHANGELOG
  tables could be swallowed as an HTML tag by some renderers — now referenced
  as a `ragnite-context` block, with the full example kept in a fenced block
  in docs/invoke-mode.md.
- docs: the absolute-interpreter examples showed a `<python>` placeholder that
  rendered confusingly — now `/path/to/python -m ragnite.cli ...`.
- Release workflow: the GitHub Release is created from the tag even when PyPI
  trusted publishing isn't configured yet (PyPI step is non-blocking until
  then).

### Added
- README: tagline ("Ragnite makes Claude Code remember before it reasons
  again.") and a 30-second TL;DR quickstart at the top.

## [0.3.0] - 2026-06-12

### Added — Invoke Mode for Claude Code (event-driven live context injection)
- `ragnite claude install`: wires the `/ragnite` slash skill
  (`.claude/skills/ragnite/SKILL.md`), the ragnite MCP server (`.mcp.json`),
  session hooks (`.claude/settings.local.json` — merged idempotently, existing
  hooks/permissions untouched), `.ragnite/config.toml` and
  `.ragnite/session.json`. Hook/MCP commands use the absolute interpreter.
- `/ragnite` skill: `init`, `invoke`, `pause`, `status`, `recall`, `remember`,
  `forget` — plus a behavior contract for obeying injected context modes.
- `ragnite claude init` (bootstrap): detects the repo root, indexes code
  (`.ragniteignore`-aware) and README/docs/configs (redacted), seeds initial
  memories — **inferences are marked `inferred=true` with reduced authority,
  never definitive** — runs a smoke recall, prints stats. Re-runs replace
  inferred records instead of duplicating.
- `ragnite claude invoke|pause|status`: session state in
  `.ragnite/session.json`, install validation, briefing injection.
- Hook handlers (`ragnite claude hook <event>`), all fail-safe (log to
  `.ragnite/hooks.log`, never break a session):
  - **SessionStart**: project briefing (brief, active decisions, constraints,
    counts); on `source=compact`, captures the compaction summary as a
    *candidate* episode before re-grounding.
  - **UserPromptSubmit**: recall on the (redacted) prompt → injects a
    `ragnite-context` XML-style block (mode, confidence, suggestion,
    evidence, sources); silent on `refuse_guess`/low confidence.
  - **PreToolUse** (Grep|Glob): advisory by default — never blocks; opt-in
    `strict` mode denies broad searches that memory answers `direct`.
  - **PostToolUse** (Edit/Write/...): incremental Code Memory re-index of the
    changed file + semantic-cache invalidation. (Bash): learns candidate
    episodes from test results and failing commands, superseding repeats.
- `CodeMemory.index_file()` (single-file incremental re-index) and
  `index_repo(ignore=...)`; loaders now also skip `vendor/` and `.claude/`.
- Security layer (`ragnite.claude.redact`): secret redaction (API keys, AWS,
  GitHub/GitLab/Slack tokens, JWTs, bearer headers, private-key blocks,
  credential assignments) applied to everything stored from live sessions;
  sensitive files (`.env*`, keys, credentials) never ingested;
  `.ragniteignore` support.
- Docs: `claude-code.md`, `invoke-mode.md`, `hooks.md`, `security.md`.
- 30 new tests (103 total): installer merge safety/idempotency, prompt
  injection contract, episodic learning + supersession, incremental re-index
  + cache invalidation, strict-mode denial, redaction, bootstrap seeding.

## [0.2.1] - 2026-06-12

Hardening pass: consistency, claim accuracy, coverage, demonstrability.

### Fixed
- FastAPI app version now follows the package version (was hardcoded `0.1.0`).
- CLI help repositioned: "Confidence-aware RAG memory engine for LLMs and
  coding agents" (was the pre-0.2 document-RAG pitch).
- **Semantic cache claim corrected**: the verdict cache saves
  retrieval/scoring/packing and reuses the packed context — it does *not* by
  itself save generation tokens. Docs and README now state the exact contract.
- HTTP `/v1/ingest` honors client-provided document `id`s (stable ids for
  eval datasets and upsert-by-id).

### Added
- **`AnswerCache`** (opt-in, `RAGNITE_ANSWER_CACHE=1`): caches *final
  generated answers* for `RagEngine.ask`/`ask_stream` — a hit is genuinely
  zero LLM tokens (`Answer.cached`). Stricter threshold (0.93) and shorter
  TTL (3d) than the verdict cache; invalidated on ingest/clear.
- 13 new tests (73 total): FastAPI version/auth/memory endpoints, MCP `recall`
  JSON contract (`recall_payload`), answer-cache LLM-skip + invalidation,
  verdict-cache invalidation on `index_repo`/`forget`, conflict→supersession
  resolution flow, incremental indexing against Ragnite's own source tree.
- `benchmarks/bench.py`: offline micro-benchmarks (cold vs cached recall,
  code-indexing time, retrieval-quality fixture).
- Docs: `agent-loop.md`, `confidence-policy.md`, `semantic-cache.md`,
  `code-memory.md`; README gained "What Ragnite is not", "When not to use",
  a before/after Claude Code session, and measured benchmark numbers.
- `.github/repo-metadata.md`: recommended GitHub description, topics and
  social-preview text.

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
