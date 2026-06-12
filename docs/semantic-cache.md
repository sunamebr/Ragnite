# Semantic Caching — exactly what is and isn't saved

Ragnite ships **two** caches with different promises. Conflating them is the
fastest way to overclaim, so here is the precise contract.

## SemanticCache (verdict cache) — always on for `MemoryEngine.recall`

Stores the **recall verdict**: packed context, mode, confidence, signals.

| Saved on a hit | NOT saved |
|---|---|
| Query embedding lookup is the only work done | — |
| Retrieval (dense + BM25) | The LLM call the *host* makes afterwards |
| Confidence scoring + mode decision | |
| Context packing | |

A hit means the agent gets the same packed context and verdict instantly. If
the host then sends that context to a model, those generation tokens are still
spent — what the host saves is *re-analysis* (and, vs. no memory at all, the
much larger cost of re-reading sources).

- Keyed by query embedding; hit threshold `RAGNITE_CACHE_THRESHOLD` (0.90).
- TTL `RAGNITE_CACHE_TTL_DAYS` (7).
- Only `direct` / `cautious` verdicts are cached — uncertainty is recomputed.
- **Cleared on every memory write** (`remember`, `forget`, `index_repo` with
  changes): new knowledge may change a verdict.
- No embedder → normalized exact-match fallback.

## AnswerCache (final-answer cache) — opt-in for `RagEngine.ask`

Stores the **finished, generated `Answer`** (text + citations). A hit returns
it without calling the LLM: **this is the only genuinely zero-LLM-token path.**

```bash
RAGNITE_ANSWER_CACHE=1 ragnite serve     # or:
```

```python
from ragnite import RagEngine, AnswerCache
engine = RagEngine(store=..., embedder=emb, llm=llm,
                   answer_cache=AnswerCache(embedder=emb, path=".ragnite/answer_cache"))
answer = await engine.ask("How does billing work?")
answer.cached   # True on a hit — no LLM call happened
```

Deliberately stricter defaults than the verdict cache, because serving a wrong
*final answer* is worse than re-running recall:

| Knob | Verdict cache | Answer cache |
|---|---:|---:|
| similarity threshold | 0.90 | **0.93** |
| TTL | 7 days | **3 days** |
| default | on | **off (opt-in)** |

- Cleared on every ingest and on `engine.clear()` — the corpus changed.
- Skipped automatically when `history` is passed (conversational context makes
  the question non-cacheable).
- `ask_stream` serves a hit as one delta + the final answer event.

## Invalidation truth table

| Event | Verdict cache | Answer cache |
|---|---|---|
| `remember` / `forget` / supersede | cleared | — |
| `index_repo` with changes | cleared | — |
| `ingest_*` documents | — | cleared |
| `engine.clear()` | — | cleared |
| TTL expiry | entry ignored | entry ignored |

Clearing whole caches on write is intentionally conservative for v0.x —
correctness over reuse. Scoped invalidation is on the roadmap.
