<div align="center">

# 🔥 Ragnite

**Confidence-Aware RAG Memory Engine for LLMs and coding agents.**

Typed memory · confidence scoring · answer modes · token-budgeted context packing · semantic cache · hybrid retrieval · MCP server

[![CI](https://github.com/sunamebr/Ragnite/actions/workflows/ci.yml/badge.svg)](https://github.com/sunamebr/Ragnite/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

---

## The problem

Every session, your coding agent re-reads the repo. Re-derives the architecture. Re-discovers that the deploy runs on Fridays, that you chose gRPC over REST six months ago, that the flaky auth test was fixed by freezing the clock. **Tokens burned re-analyzing what should already be consolidated** — and worse, the agent answers with the same false confidence whether it knows or guesses.

Ragnite is the missing layer: a **memory and conviction engine**. Agents write consolidated knowledge once; on every question, one `recall()` returns the *smallest sufficient context* plus an explicit verdict on how much to trust it.

```python
from ragnite import build_memory_engine

memory = build_memory_engine()

await memory.remember_decision("Services communicate over gRPC.", subject="api-style")
await memory.remember_fact("The database listens on port 5432.", subject="db-port")
await memory.remember_episode("Fixed flaky auth test by freezing the clock.")
await memory.index_repo(".")          # Code Memory: files, symbols, endpoints, deps

answer = await memory.recall("how do services talk to each other?")
answer.mode        # "direct" — strong consolidated evidence
answer.confidence  # 0.86
answer.context     # token-budgeted evidence pack, ready to inject
answer.suggestion  # "Answer directly from the provided context — do not re-analyze."
```

## Architecture

```mermaid
flowchart TB
    subgraph Memories["Typed Memory Bank"]
        F["📌 Factual Memory<br/>stable project/domain truths"]
        D["🏛️ Decision Memory<br/>architectural choices, supersession chains"]
        E["📓 Episodic Memory<br/>bugs fixed, failed attempts, progress"]
        C["🧩 Code Memory<br/>files, symbols, endpoints, deps, tests"]
    end
    Q[query] --> SC{Semantic Cache<br/>equivalent question?}
    SC -- hit --> OUT
    SC -- miss --> R[Hybrid Recall<br/>dense + BM25]
    Memories --> R
    R --> CS[Confidence Scorer<br/>similarity · sources · agreement<br/>recency · authority · conflicts]
    CS --> AM{Answer Mode}
    AM --> CP[Context Packer<br/>smallest context under token budget]
    CP --> OUT["MemoryAnswer<br/>mode + confidence + context + suggestion"]
    OUT -. cache .-> SC
```

### Answer modes — the conviction contract

| Mode | Meaning | What the agent should do |
|---|---|---|
| `direct` | Strong consolidated evidence | Answer from context; **do not re-analyze** |
| `cautious` | Partial evidence | Answer with explicit caveats and attribution |
| `ask_clarification` | Conflicting/ambiguous memory entries | Ask the user one targeted question |
| `search_more` | Weak evidence | Retrieve more (code search, docs, web) first |
| `refuse_guess` | No reliable basis | Say "I don't know" instead of hallucinating |

### Confidence Scorer

Confidence blends seven signals — **top similarity, mean similarity, source count, dense↔keyword agreement, recency (per-kind half-life), source authority, conflict detection** — and is hard-capped by relevance: fresh, authoritative sources can never make a weak match look trustworthy. Conflicts (two active entries claiming the same `subject`) force `ask_clarification` until a decision supersedes the loser.

### Context Packer

Greedy value-ordered packing under a token budget (default 2000), near-duplicate suppression, compact one-line headers with kind/similarity/age/provenance. The LLM gets evidence, not prose.

### Semantic Cache

Queries are cached by embedding; an equivalent question returns the previous verdict + context with **zero retrieval, zero scoring, zero LLM tokens**. TTL-bounded, invalidated on memory writes.

### Code Memory

`index_repo()` parses the repository into memory: Python via AST (functions, classes, methods, docstrings, imports, FastAPI/Flask-style routes), other languages via definition-boundary extraction. Incremental — unchanged files are hash-skipped, deleted files evicted. "Where is auth handled?" becomes one recall instead of a directory crawl.

## Plug it into Claude Code (MCP)

```bash
pip install "ragnite[mcp,anthropic]"
claude mcp add ragnite -- ragnite mcp
```

Tools exposed: `recall` (verdict + packed context), `remember`, `remember_decision` (with supersession), `index_repo`, `forget`, plus document RAG (`search`, `ask`, `ingest_*`) and `stats`. Claude Desktop config:

```json
{
  "mcpServers": {
    "ragnite": {
      "command": "ragnite",
      "args": ["mcp"],
      "env": { "VOYAGE_API_KEY": "...", "ANTHROPIC_API_KEY": "..." }
    }
  }
}
```

The intended agent loop: **`recall` before re-reading anything; obey the mode; `remember` whatever was expensive to figure out.**

## Install & quickstart

```bash
pip install ragnite                  # core — runs offline (BM25 + native store)
pip install "ragnite[all]"           # + Claude, server, MCP, Qdrant, PDF/DOCX
```

```bash
# memory & conviction
ragnite index-code .
ragnite remember "We deploy Fridays at noon" --kind decision --subject deploy-window
ragnite recall "when do we deploy?"
#  mode: direct  confidence: 0.84  tokens: 31
#  - [decision|sim 0.79|today] deploy-window: We deploy Fridays at noon

# document RAG (grounded answers with [n] citations)
ragnite ingest ./docs
ragnite ask "how does billing work?"
```

Embeddings: Voyage AI (recommended with Claude), any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, Jina), or local sentence-transformers. Generation: Claude via the official SDK (default `claude-opus-4-8`) or OpenAI-compatible. **No keys at all? Everything still works on BM25.**

## Document RAG (the second half)

The classic pipeline is still here and production-grade: loaders (md/code/html/json/pdf/docx) → recursive/markdown/code-aware chunking → optional **contextual retrieval** (Anthropic technique, prompt-cached) → hybrid dense+BM25 with **RRF fusion** → optional reranking (Cohere/Voyage/LLM listwise) → **grounded answers with structured citations**, sync or SSE streaming.

```python
from ragnite import build_engine

engine = build_engine()
await engine.ingest_path("./docs")
result = await engine.ask("How does billing work?")   # result.citations -> [Citation(...)]
```

## HTTP API

```bash
pip install "ragnite[server]" && ragnite serve
```

| Method | Route | Description |
|---|---|---|
| `POST` | `/v1/memory/recall` | `{"query"}` → mode + confidence + packed context |
| `POST` | `/v1/memory/remember` | `{"text", "kind", "subject?", "supersedes?"}` |
| `POST` | `/v1/memory/index_code` | `{"path"}` → incremental code indexing |
| `GET` | `/v1/memory/stats` | records by kind, cache entries, policy |
| `POST` | `/v1/ingest` / `/v1/search` / `/v1/ask` | document RAG (SSE on `stream: true`) |
| `GET` | `/healthz` / `/v1/stats` | ops |

Set `RAGNITE_API_KEY` for bearer auth.

## Who is this for

Coding agents (Claude Code, autonomous loops) · LLM assistants · MCP servers · living documentation · project memory · semantic search · engineering teams that want decisions and tribal knowledge queryable with a confidence score.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `RAGNITE_EMBEDDER` | `auto` | `voyage` \| `openai` \| `local` \| `fake` \| `none` (auto-detects by key) |
| `RAGNITE_LLM` / `RAGNITE_LLM_MODEL` | `auto` / `claude-opus-4-8` | `anthropic` \| `openai` \| `none` |
| `RAGNITE_STORE` | `native` | `qdrant` for scale-out (docs **and** memory bank) |
| `RAGNITE_MEMORY_BUDGET` | `2000` | context-packer token budget |
| `RAGNITE_CACHE_THRESHOLD` / `_TTL_DAYS` | `0.90` / `7` | semantic cache similarity & freshness |
| `RAGNITE_RERANKER` | `none` | `cohere` \| `voyage` \| `llm` |
| `RAGNITE_CONTEXTUAL` | `0` | contextual retrieval at ingest |
| `RAGNITE_DATA_DIR` | `.ragnite` | bank, semcache, doc collections, embedding cache |

Full list in [.env.example](.env.example). Confidence thresholds/weights are code-level: `ConfidencePolicy(direct_threshold=..., w_recency=...)`.

## Scaling

Native NumPy store (exact cosine, persisted) handles hundreds of thousands of records with zero infra; `RAGNITE_STORE=qdrant` moves both document collections and the memory bank to Qdrant for sharding/HA. `docker compose up` in [docker/](docker) ships API + Qdrant. Embedding cache (SQLite) makes re-indexing free; eval suite (`ragnite eval`, hit@k/MRR/nDCG + LLM-judge) keeps retrieval quality regression-tested in CI.

## Project layout

```
src/ragnite/
├── memory/      ★ the conviction layer
│   ├── bank.py        typed memory bank (fact/decision/episode/code)
│   ├── scorer.py      confidence signals + answer-mode policy
│   ├── packer.py      token-budgeted context assembly
│   ├── semcache.py    semantic answer cache
│   ├── code_index.py  incremental repository indexing
│   └── engine.py      MemoryEngine.recall() -> MemoryAnswer
├── ingest/      loaders + chunkers
├── embed/       Voyage / OpenAI-compat / local / fake + SQLite cache
├── store/       native NumPy store, Qdrant adapter
├── retrieve/    BM25, RRF fusion, rerankers
├── llm/         Anthropic (official SDK), OpenAI-compatible
├── rag/         document RAG engine, contextual retrieval, prompts
├── eval/        IR metrics + LLM-judge
└── server/      FastAPI + MCP
```

## Roadmap

- [ ] Auto-consolidation: distill episodes into facts/decisions on a schedule
- [ ] LLM-assisted conflict resolution and memory dedup
- [ ] Memory decay & promotion policies (episode → fact)
- [ ] pgvector / Milvus store adapters; GraphRAG-style entity linking
- [ ] Multi-tenant memory banks with per-tenant auth
- [ ] OpenTelemetry tracing; token-savings analytics per agent

## Contributing & license

PRs welcome — [CONTRIBUTING.md](CONTRIBUTING.md). Test suite runs fully offline: `uv sync --group dev && uv run pytest`. [MIT](LICENSE) © sunamebr
