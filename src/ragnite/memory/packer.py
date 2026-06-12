"""ContextPacker — assemble the smallest useful context under a token budget.

Greedy, value-ordered packing with near-duplicate suppression. Output is a
compact bracket-headed line per memory so the agent sees kind, confidence
signals and provenance without paying for prose.
"""

from __future__ import annotations

from ragnite.memory.types import Evidence, PackedContext
from ragnite.retrieve.bm25 import tokenize


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic — deliberately provider-agnostic."""
    return max(1, (len(text) + 3) // 4)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _age_label(days: float) -> str:
    if days < 1:
        return "today"
    if days < 60:
        return f"{int(days)}d"
    return f"{int(days // 30)}mo"


class ContextPacker:
    def __init__(
        self,
        budget_tokens: int = 2000,
        dedupe_threshold: float = 0.85,
        relative_floor: float = 0.30,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.dedupe_threshold = dedupe_threshold
        # evidence below relative_floor * top_similarity is noise, not context
        self.relative_floor = relative_floor

    def _line(self, evidence: Evidence) -> str:
        record = evidence.record
        parts = [record.kind.value, f"sim {evidence.similarity:.2f}", _age_label(record.age_days)]
        if record.source:
            parts.append(record.source)
        header = "|".join(parts)
        subject = f"{record.subject}: " if record.subject else ""
        return f"- [{header}] {subject}{record.text}"

    def pack(self, evidence: list[Evidence], budget_tokens: int | None = None) -> PackedContext:
        budget = budget_tokens or self.budget_tokens
        lines: list[str] = []
        token_sets: list[set[str]] = []
        used = 0
        spent = 0
        truncated = False
        floor = evidence[0].similarity * self.relative_floor if evidence else 0.0

        for ev in evidence:
            if ev.similarity < floor and used > 0:
                break  # remaining evidence is noise relative to the best match
            tokens_in_text = set(tokenize(ev.record.text))
            if any(_jaccard(tokens_in_text, seen) >= self.dedupe_threshold for seen in token_sets):
                continue  # near-duplicate of something already packed
            line = self._line(ev)
            cost = estimate_tokens(line)
            if spent + cost > budget:
                if used == 0:
                    # always carry the single best evidence, trimmed to fit
                    max_chars = max(80, budget * 4 - 16)
                    line = line[:max_chars]
                    cost = estimate_tokens(line)
                    truncated = True
                else:
                    truncated = True
                    continue
            lines.append(line)
            token_sets.append(tokens_in_text)
            used += 1
            spent += cost

        return PackedContext(text="\n".join(lines), used=used, tokens=spent, truncated=truncated)
