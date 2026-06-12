import pytest

from ragnite.embed.fake import FakeEmbedder
from ragnite.memory.bank import MemoryBank
from ragnite.memory.engine import MemoryEngine
from ragnite.memory.semcache import SemanticCache
from ragnite.memory.types import MemoryKind


@pytest.fixture
def memory(tmp_path) -> MemoryEngine:
    return MemoryEngine(
        bank=MemoryBank(embedder=FakeEmbedder(), path=tmp_path / "bank"),
        cache=SemanticCache(embedder=FakeEmbedder(), path=tmp_path / "semcache"),
    )


async def test_empty_memory_refuses_to_guess(memory):
    answer = await memory.recall("what is the deploy schedule?")
    assert answer.mode == "refuse_guess"
    assert answer.confidence == 0.0
    assert answer.context == ""
    assert "know" in answer.suggestion


async def test_consolidated_facts_answer_with_conviction(memory):
    await memory.remember_fact("The deploy pipeline runs on GitHub Actions.", subject="deploy-ci")
    await memory.remember_fact("The deploy pipeline requires green tests.", subject="deploy-gate")
    await memory.remember_fact("Deploys happen every Friday at noon.", subject="deploy-window")

    answer = await memory.recall("what runs the deploy pipeline?", use_cache=False)
    assert answer.mode in {"direct", "cautious"}
    assert answer.confidence >= 0.5
    assert "GitHub Actions" in answer.context
    assert answer.evidence
    assert answer.tokens > 0
    assert "do not re-derive" in answer.suggestion or "caveats" in answer.suggestion


async def test_semantic_cache_short_circuits_second_call(memory):
    await memory.remember_fact("The deploy pipeline runs on GitHub Actions.", subject="deploy-ci")
    first = await memory.recall("what runs the deploy pipeline?")
    assert first.cached is False
    second = await memory.recall("what runs the deploy pipeline?")
    assert second.cached is True
    assert second.mode == first.mode
    assert second.context == first.context


async def test_remember_invalidates_cache(memory):
    await memory.remember_fact("The database listens on port 5432.", subject="db-port")
    await memory.recall("which port does the database use?")
    await memory.remember_fact("Background workers poll the queue.", subject="workers")
    fresh = await memory.recall("which port does the database use?")
    assert fresh.cached is False  # cache was cleared by the write


async def test_conflicting_facts_ask_for_clarification(memory):
    await memory.remember_fact("The database listens on port 5432.", subject="db-port")
    await memory.remember_fact("The database listens on port 5433.", subject="db-port")
    answer = await memory.recall("which port does the database use?", use_cache=False)
    assert answer.signals.conflict is True
    assert answer.mode == "ask_clarification"


async def test_decision_supersedes_retires_old_entry(memory):
    old = await memory.remember_decision("API style: REST with JSON.", subject="api-style")
    new = await memory.remember_decision(
        "API style: gRPC between services.", subject="api-style", supersedes=old.id
    )
    answer = await memory.recall("which api style do we use?", use_cache=False)
    ids = [evidence.record.id for evidence in answer.evidence]
    assert new.id in ids
    assert old.id not in ids  # superseded entries never come back
    assert answer.mode != "ask_clarification"


async def test_kind_filter_and_stats(memory):
    await memory.remember_fact("Project codename is Ragnite.", subject="codename")
    await memory.remember_episode("Fixed flaky auth test by freezing the clock.", subject="auth-test")
    only_episodes = await memory.recall("auth test fix", kinds=[MemoryKind.EPISODE], use_cache=False)
    assert all(e.record.kind is MemoryKind.EPISODE for e in only_episodes.evidence)

    stats = await memory.stats()
    assert stats["active_by_kind"]["fact"] == 1
    assert stats["active_by_kind"]["episode"] == 1
    assert stats["records"] == 2


async def test_conflict_resolved_by_supersession_and_forget(memory):
    a = await memory.remember_fact("The database listens on port 5432.", subject="db-port")
    b = await memory.remember_fact("The database listens on port 5433.", subject="db-port")
    conflicted = await memory.recall("which port does the database use?", use_cache=False)
    assert conflicted.mode == "ask_clarification"

    # the agent asks, the user arbitrates: retire the loser, supersede the rest
    await memory.forget(b.id)
    await memory.remember_decision(
        "The database listens on port 5432 (confirmed).", subject="db-port", supersedes=a.id
    )
    resolved = await memory.recall("which port does the database use?", use_cache=False)
    assert resolved.signals.conflict is False
    assert resolved.mode != "ask_clarification"


async def test_index_repo_and_forget_invalidate_cache(memory, tmp_path):
    await memory.remember_fact("The deploy pipeline runs on GitHub Actions.", subject="deploy-ci")
    warm = await memory.recall("what runs the deploy pipeline?")
    assert warm.cached is False
    assert (await memory.recall("what runs the deploy pipeline?")).cached is True

    repo = tmp_path / "tiny-repo"
    repo.mkdir()
    (repo / "main.py").write_text("def run():\n    pass\n", encoding="utf-8")
    await memory.index_repo(repo)
    assert (await memory.recall("what runs the deploy pipeline?")).cached is False

    record = await memory.remember_fact("Temp fact.", subject="tmp")
    await memory.recall("temp fact?")
    await memory.forget(record.id)
    assert (await memory.recall("temp fact?")).cached is False


async def test_forget(memory):
    record = await memory.remember_fact("Temporary wrong fact.", subject="oops")
    assert await memory.forget(record.id) is True
    answer = await memory.recall("temporary wrong fact", use_cache=False)
    assert all(e.record.id != record.id for e in answer.evidence)
