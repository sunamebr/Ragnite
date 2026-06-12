"""Confidence scoring and answer-mode policy.

Confidence is a weighted blend of seven signals — similarity, source count,
dense/keyword agreement, recency, source authority, conflicts — and is hard-
capped by relevance: weak similarity can never produce high confidence no
matter how fresh or authoritative the sources are.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from ragnite.memory.types import (
    AnswerMode,
    ConfidenceReport,
    ConfidenceSignals,
    Evidence,
    MemoryKind,
)


class ConfidencePolicy(BaseModel):
    # answer-mode thresholds on the final score
    direct_threshold: float = 0.70
    cautious_threshold: float = 0.50
    search_threshold: float = 0.20
    # what counts as a "strong" similarity match
    strong_similarity: float = 0.35
    # signal weights (sum to 1.0)
    w_similarity: float = 0.30
    w_sources: float = 0.15
    w_agreement: float = 0.15
    w_recency: float = 0.10
    w_authority: float = 0.15
    w_no_conflict: float = 0.15
    # sources needed to saturate the source-count signal
    max_sources: int = 3
    # freshness half-life per memory kind (days)
    half_life_days: dict[str, float] = Field(
        default_factory=lambda: {
            MemoryKind.FACT.value: 180.0,
            MemoryKind.DECISION.value: 365.0,
            MemoryKind.EPISODE.value: 30.0,
            MemoryKind.CODE.value: 21.0,
        }
    )


def _detect_conflict(evidence: list[Evidence]) -> list[str]:
    """Same active subject claimed by unlinked fact/decision records = conflict."""
    seen: dict[str, Evidence] = {}
    conflicted: list[str] = []
    for ev in evidence[:6]:
        record = ev.record
        if record.kind not in (MemoryKind.FACT, MemoryKind.DECISION) or not record.subject:
            continue
        subject = record.subject.lower()
        other = seen.get(subject)
        if other is None:
            seen[subject] = ev
            continue
        linked = record.supersedes == other.record.id or other.record.supersedes == record.id
        if other.record.id != record.id and not linked:
            conflicted.append(subject)
    return conflicted


class ConfidenceScorer:
    def __init__(self, policy: ConfidencePolicy | None = None) -> None:
        self.policy = policy or ConfidencePolicy()

    def _recency(self, evidence: Evidence) -> float:
        half_life = self.policy.half_life_days.get(evidence.record.kind.value, 90.0)
        return math.pow(0.5, evidence.record.age_days / max(half_life, 1.0))

    def score(self, evidence: list[Evidence]) -> ConfidenceReport:
        p = self.policy
        if not evidence:
            return ConfidenceReport(
                score=0.0, signals=ConfidenceSignals(), rationale=["no evidence found in memory"]
            )

        top = evidence[: p.max_sources]
        top_sim = evidence[0].similarity
        mean_sim = sum(e.similarity for e in top) / len(top)
        strong = [e for e in evidence if e.similarity >= p.strong_similarity]
        source_signal = min(1.0, len(strong) / p.max_sources)

        dense_used = any(e.in_dense for e in evidence)
        bm25_used = any(e.in_bm25 for e in evidence)
        if dense_used and bm25_used:
            head = evidence[: min(4, len(evidence))]
            agreement = sum(1 for e in head if e.in_dense and e.in_bm25) / len(head)
        else:
            agreement = 0.5  # only one retriever ran — neutral, not damning

        recency = sum(self._recency(e) for e in top) / len(top)
        authority = sum(e.record.authority for e in top) / len(top)
        conflicts = _detect_conflict(evidence)

        raw = (
            p.w_similarity * min(1.0, top_sim)
            + p.w_sources * source_signal
            + p.w_agreement * agreement
            + p.w_recency * recency
            + p.w_authority * authority
            + p.w_no_conflict * (0.0 if conflicts else 1.0)
        )
        # relevance cap: confidence can never outrun similarity
        relevance = min(1.0, top_sim / p.strong_similarity)
        score = round(raw * relevance, 4)

        rationale = [
            f"top similarity {top_sim:.2f} across {len(evidence)} candidate(s)",
            f"{len(strong)} strong source(s); retriever agreement {agreement:.2f}",
            f"recency {recency:.2f}; authority {authority:.2f}",
        ]
        if conflicts:
            rationale.append(f"conflicting active entries on: {', '.join(sorted(set(conflicts)))}")

        signals = ConfidenceSignals(
            top_similarity=round(top_sim, 4),
            mean_similarity=round(mean_sim, 4),
            source_count=round(source_signal, 4),
            agreement=round(agreement, 4),
            recency=round(recency, 4),
            authority=round(authority, 4),
            conflict=bool(conflicts),
        )
        return ConfidenceReport(score=score, signals=signals, rationale=rationale)


def decide_mode(report: ConfidenceReport, policy: ConfidencePolicy) -> AnswerMode:
    if report.signals.conflict:
        return "ask_clarification"
    score = report.score
    if score >= policy.direct_threshold:
        return "direct"
    if score >= policy.cautious_threshold:
        return "cautious"
    if score >= policy.search_threshold:
        return "search_more"
    return "refuse_guess"
