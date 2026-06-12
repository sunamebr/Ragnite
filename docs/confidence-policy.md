# Confidence Policy — signals, formula, tuning

Everything the `ConfidenceScorer` does is inspectable (`answer.signals`,
`report.rationale`) and tunable (`ConfidencePolicy`). No magic.

## The formula

```
raw   = w_similarity  · min(1, top_similarity)
      + w_sources     · min(1, strong_sources / max_sources)
      + w_agreement   · agreement
      + w_recency     · recency
      + w_authority   · authority
      + w_no_conflict · (0 if conflict else 1)

score = raw × min(1, top_similarity / strong_similarity)     # relevance cap
```

The **relevance cap** is the load-bearing design decision: authority, recency
and corroboration are *multipliers of trust in a match*, never substitutes for
one. A barely-similar record from a perfect source stays low-confidence.

## Signals

| Signal | Default weight | Computed from |
|---|---:|---|
| `top_similarity` | 0.30 | best evidence similarity (cosine, or squashed BM25 `s/(s+4)`) |
| `source_count` | 0.15 | records with similarity ≥ `strong_similarity` (0.35), saturating at `max_sources` (3) |
| `agreement` | 0.15 | fraction of the head found by **both** dense and BM25; neutral 0.5 when only one retriever ran |
| `recency` | 0.10 | `0.5 ** (age_days / half_life[kind])`, averaged over top-3 |
| `authority` | 0.15 | mean record authority (kind-defaulted: decision 0.9, fact 0.8, code 0.7, episode 0.6) |
| `no_conflict` | 0.15 | 1 unless unlinked active records share a `subject` |

Half-lives per kind: fact 180d · decision 365d · episode 30d · code 21d.

## Answer-mode thresholds

```
conflict detected            -> ask_clarification   (always)
score >= 0.70 (direct)       -> direct
score >= 0.50 (cautious)     -> cautious
score >= 0.20 (search)       -> search_more
otherwise / no evidence      -> refuse_guess
```

## Conflict semantics

Two **active** `fact`/`decision` records claiming the same `subject`, with no
`supersedes` link between them, is a conflict. Resolution paths:

- `remember_decision(..., supersedes=old_id)` — the old record flips to
  `superseded` and never resurfaces.
- `forget(loser_id)` — for plain wrong entries.

Subjects are the conflict-detection key: **use them**. Entries without a
subject can never conflict (and never get the protection).

## Tuning recipes

```python
from ragnite import ConfidencePolicy, ConfidenceScorer, MemoryEngine

# Stricter "direct" — for agents that act autonomously on direct answers
policy = ConfidencePolicy(direct_threshold=0.80)

# Fast-moving codebase — make code memory age faster
policy = ConfidencePolicy(half_life_days={"code": 7.0, "episode": 14.0,
                                          "fact": 180.0, "decision": 365.0})

# Trust corroboration more than raw similarity
policy = ConfidencePolicy(w_similarity=0.22, w_sources=0.23)

engine = MemoryEngine(bank, scorer=ConfidenceScorer(policy))
```

Sanity rules when tuning: keep the weights summing to ~1.0; keep
`strong_similarity` aligned with your embedder's score distribution (real
cosine models cluster higher than the hashed test embedder); and prefer moving
*thresholds* before moving *weights* — thresholds change behavior, weights
change calibration.

## Known limits (v0.2)

- Conflict detection is subject-keyed, not semantic: two contradicting entries
  with different subjects won't be flagged. LLM-assisted contradiction checks
  are on the roadmap.
- `agreement` is neutral (0.5) in BM25-only mode, so keyword-only deployments
  top out at slightly lower scores — by design, since single-retriever
  evidence is weaker.
