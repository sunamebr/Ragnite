from ragnite.memory.packer import ContextPacker, estimate_tokens
from ragnite.memory.types import Evidence, MemoryKind, MemoryRecord


def _evidence(text: str, similarity: float = 0.8, source: str | None = None) -> Evidence:
    return Evidence(
        record=MemoryRecord(kind=MemoryKind.FACT, text=text, source=source),
        similarity=similarity,
    )


def test_budget_is_respected():
    evidence = [
        _evidence(f"unique entry number {i} " + "filler word " * 30, 0.9 - i * 0.01) for i in range(10)
    ]
    packed = ContextPacker().pack(evidence, budget_tokens=60)
    assert packed.tokens <= 60
    assert packed.truncated is True
    assert 1 <= packed.used < 10


def test_best_evidence_comes_first_with_provenance():
    evidence = [
        _evidence("postgres sixteen is the database", 0.9, source="docs/adr/001.md"),
        _evidence("the frontend uses react", 0.4),
    ]
    packed = ContextPacker().pack(evidence)
    first_line = packed.text.splitlines()[0]
    assert "postgres" in first_line
    assert "docs/adr/001.md" in first_line
    assert "[fact|" in first_line


def test_near_duplicates_are_skipped():
    evidence = [
        _evidence("the deploy runs every friday at noon", 0.9),
        _evidence("the deploy runs every friday at noon", 0.85),  # duplicate
        _evidence("staging mirrors production data weekly", 0.7),
    ]
    packed = ContextPacker().pack(evidence)
    assert packed.used == 2


def test_single_oversized_evidence_is_trimmed_in():
    evidence = [_evidence("x" * 4000, 0.9)]
    packed = ContextPacker().pack(evidence, budget_tokens=100)
    assert packed.used == 1
    assert packed.truncated is True
    assert estimate_tokens(packed.text) <= 110
