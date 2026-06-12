import time

from ragnite.memory.scorer import ConfidencePolicy, ConfidenceScorer, decide_mode
from ragnite.memory.types import Evidence, MemoryKind, MemoryRecord


def _evidence(
    similarity: float,
    kind: MemoryKind = MemoryKind.FACT,
    subject: str | None = None,
    age_days: float = 0.0,
    authority: float = 0.8,
    in_dense: bool = True,
    in_bm25: bool = True,
    supersedes: str | None = None,
) -> Evidence:
    stamp = time.time() - age_days * 86400
    record = MemoryRecord(
        kind=kind,
        text=f"evidence with similarity {similarity}",
        subject=subject,
        authority=authority,
        supersedes=supersedes,
        created_at=stamp,
        updated_at=stamp,
    )
    return Evidence(record=record, similarity=similarity, in_dense=in_dense, in_bm25=in_bm25)


scorer = ConfidenceScorer()
policy = scorer.policy


def test_no_evidence_refuses():
    report = scorer.score([])
    assert report.score == 0.0
    assert decide_mode(report, policy) == "refuse_guess"


def test_strong_agreeing_sources_answer_direct():
    evidence = [_evidence(0.85), _evidence(0.80), _evidence(0.75)]
    report = scorer.score(evidence)
    assert report.score >= policy.direct_threshold
    assert decide_mode(report, policy) == "direct"
    assert report.signals.conflict is False


def test_weak_single_source_searches_more():
    report = scorer.score([_evidence(0.22, in_bm25=False)])
    assert policy.search_threshold <= report.score < policy.cautious_threshold
    assert decide_mode(report, policy) == "search_more"


def test_relevance_caps_confidence():
    # fresh, authoritative, agreeing — but barely similar: must not score high
    evidence = [_evidence(0.10, authority=1.0), _evidence(0.08, authority=1.0)]
    report = scorer.score(evidence)
    assert report.score < policy.cautious_threshold


def test_conflicting_subjects_ask_clarification():
    evidence = [
        _evidence(0.8, subject="db-port"),
        _evidence(0.78, subject="db-port"),
    ]
    report = scorer.score(evidence)
    assert report.signals.conflict is True
    assert decide_mode(report, policy) == "ask_clarification"


def test_supersedes_link_is_not_a_conflict():
    old = _evidence(0.7, subject="api-style")
    new = _evidence(0.8, subject="api-style", supersedes=old.record.id)
    report = scorer.score([new, old])
    assert report.signals.conflict is False


def test_recency_decay_lowers_old_episodes():
    fresh = scorer.score([_evidence(0.7, kind=MemoryKind.EPISODE, age_days=0)])
    stale = scorer.score([_evidence(0.7, kind=MemoryKind.EPISODE, age_days=120)])
    assert fresh.score > stale.score
    assert stale.signals.recency < 0.1


def test_custom_policy_thresholds():
    strict = ConfidencePolicy(direct_threshold=0.95)
    report = ConfidenceScorer(strict).score([_evidence(0.85), _evidence(0.8), _evidence(0.75)])
    assert decide_mode(report, strict) == "cautious"
