# The Memory & Conviction Layer

Ragnite's core thesis: an agent should never pay twice to learn the same
thing, and should always know *how much* to trust what it recalls.

## Memory taxonomy

| Kind | What goes in | Default authority | Recency half-life |
|---|---|---:|---:|
| `fact` | Stable truths: project, product, domain, org | 0.80 | 180d |
| `decision` | Architectural & strategic decisions taken | 0.90 | 365d |
| `episode` | Dev events: bugs fixed, failed attempts, progress | 0.60 | 30d |
| `code` | Indexed repo: files, symbols, deps, endpoints, tests | 0.70 | 21d |

Every `MemoryRecord` carries `subject` (topic key), `authority`, `status`
(`active` / `superseded` / `deprecated`), `supersedes` (decision lineage) and
timestamps. Records are stored in any `VectorStore` and indexed by BM25 in
parallel; the `subject` doubles as a contextual prefix for the embedding.

### Supersession (Decision Memory)

```python
old = await memory.remember_decision("API style: REST.", subject="api-style")
new = await memory.remember_decision("API style: gRPC.", subject="api-style",
                                     supersedes=old.id)
```

The old record flips to `superseded` and never resurfaces in recall. Two
*active* records on the same subject with no supersession link are a
**conflict** — recall then returns `ask_clarification` until a human (or the
agent, after asking) records the winning decision.

## recall() pipeline

```
query
 ├─ 1. SemanticCache.get(query)      # equivalent question seen before? return verdict, done
 ├─ 2. MemoryBank.recall(query)      # dense (cosine) + BM25, merged per record
 ├─ 3. ConfidenceScorer.score()      # 7 signals -> score + rationale
 ├─ 4. decide_mode()                 # direct | cautious | ask_clarification | search_more | refuse_guess
 ├─ 5. ContextPacker.pack()          # smallest context under token budget
 └─ 6. SemanticCache.put()           # cache direct/cautious verdicts
```

### Confidence signals

| Signal | Source | Notes |
|---|---|---|
| `top_similarity` | best dense cosine / squashed BM25 | dominant term |
| `mean_similarity` | top-3 average | reported, not weighted separately |
| `source_count` | # records above `strong_similarity`, saturating at 3 | corroboration |
| `agreement` | fraction of head results found by *both* retrievers | lexical+semantic consensus; neutral 0.5 when only one retriever ran |
| `recency` | `0.5 ** (age_days / half_life[kind])` | episodes age fast, decisions slowly |
| `authority` | mean record authority | source trust, kind-defaulted |
| `conflict` | unlinked active records sharing a subject | forces `ask_clarification` |

`score = Σ(w_i · signal_i) × min(1, top_similarity / strong_similarity)` —
the multiplicative relevance cap guarantees that freshness and authority can
never compensate for a weak match.

Tune everything via `ConfidencePolicy`:

```python
from ragnite import ConfidencePolicy, ConfidenceScorer, MemoryEngine

policy = ConfidencePolicy(direct_threshold=0.8, w_recency=0.2)
engine = MemoryEngine(bank, scorer=ConfidenceScorer(policy))
```

### Packed context format

```
- [decision|sim 0.81|3mo|docs/adr/007.md] api-style: Services communicate over gRPC.
- [fact|sim 0.64|today] db-port: The database listens on port 5432.
```

Kind, similarity, age and provenance in ~15 tokens of overhead per entry;
near-duplicates (token Jaccard ≥ 0.85) are dropped; the budget (default 2000
tokens, `RAGNITE_MEMORY_BUDGET`) is a hard cap.

## Semantic cache semantics

The verdict cache stores the recall verdict (context + mode + signals), **not
a final LLM answer** — a hit saves retrieval/scoring/packing; generation
tokens are saved only by the opt-in `AnswerCache` on the document-RAG side.
Exact promises and the invalidation truth table:
[semantic-cache.md](semantic-cache.md).

- Keyed by **query embedding**; hit threshold `RAGNITE_CACHE_THRESHOLD` (0.90).
- TTL `RAGNITE_CACHE_TTL_DAYS` (7).
- Only `direct` and `cautious` verdicts are cached — uncertainty is recomputed.
- **Any memory write clears the cache** (correctness over reuse: new knowledge
  may change a verdict).
- No embedder configured → degrades to normalized exact-match.

## Code Memory

`index_repo(path)` produces, per file: one structural *file record* (language,
imports, symbol roster) and one record per symbol (functions, classes,
methods) with signature + docstring head, `file:line` provenance, `endpoint`
tag + route for decorator-based HTTP handlers, and `test` tags. Python uses
`ast`; other languages a definition-boundary regex. Indexing is incremental by
content hash; deleted files are evicted. `CodeMemory.graph()` returns the
file→imports relation map.

## The agent contract

A well-behaved agent (system prompt or MCP tool description handles this):

1. **Start of session:** `index_repo(".")` (cheap — unchanged files skip).
2. **Before re-reading or re-deriving anything:** `recall(question)`.
3. **Obey the mode** — `direct` means *do not re-analyze*; `search_more` means
   the memory layer itself is telling you to go look.
4. **After figuring out anything expensive:** `remember(...)` with a good
   `subject`; decisions that replace old ones pass `supersedes`.

That loop is the token-savings flywheel: each session makes the next cheaper.
