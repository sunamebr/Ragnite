"""Invoke-mode hook handlers, exercised as pure functions (no subprocess)."""

import asyncio
import json

import pytest

from ragnite.claude.bootstrap import project_config, run_init
from ragnite.claude.hooks import (
    handle_post_tool,
    handle_pre_tool,
    handle_session_start,
    handle_user_prompt,
    run_hook,
)
from ragnite.claude.session import SessionState
from ragnite.config import build_memory_engine
from ragnite.memory.types import MemoryKind


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGNITE_EMBEDDER", "fake")
    monkeypatch.setenv("RAGNITE_LLM", "none")
    root = tmp_path / "proj"
    (root / ".ragnite").mkdir(parents=True)
    SessionState(root).set_active(True)
    return root


def _engine(root):
    return build_memory_engine(project_config(root))


def _seed_facts(root, *facts: tuple[str, str]):
    engine = _engine(root)
    for text, subject in facts:
        asyncio.run(engine.remember_fact(text, subject=subject))


def _payload(root, **extra) -> dict:
    return {"cwd": str(root), "session_id": "s1", **extra}


# -- UserPromptSubmit ---------------------------------------------------------------


def test_user_prompt_inactive_returns_none(project):
    SessionState(project).set_active(False)
    assert handle_user_prompt(_payload(project, prompt="which port does the database use?")) is None


def test_user_prompt_injects_ragnite_context(project):
    _seed_facts(project, ("The database listens on port 5432.", "db-port"))
    result = handle_user_prompt(_payload(project, prompt="which port does the database use?"))
    assert result is not None
    context = result["hookSpecificOutput"]["additionalContext"]
    assert result["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "<ragnite-context mode=" in context
    assert "5432" in context
    assert "suggestion:" in context
    assert SessionState(project).data["stats"]["prompts_enriched"] == 1


def test_user_prompt_skips_slash_commands_short_prompts_and_unknowns(project):
    assert handle_user_prompt(_payload(project, prompt="/ragnite status please")) is None
    assert handle_user_prompt(_payload(project, prompt="ok")) is None
    # empty memory -> refuse_guess -> nothing injected
    assert handle_user_prompt(_payload(project, prompt="completely unknown topic xyzzy?")) is None


# -- SessionStart ---------------------------------------------------------------------


def test_session_start_briefing_lists_decisions(project):
    engine = _engine(project)
    asyncio.run(engine.remember_decision("Services communicate over gRPC.", subject="api-style"))
    result = handle_session_start(_payload(project, source="startup"))
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "RAGNITE PROJECT MEMORY" in context
    assert "gRPC" in context
    assert "recall" in context


def test_session_start_inactive_or_empty(project, tmp_path):
    SessionState(project).set_active(False)
    assert handle_session_start(_payload(project)) is None


def test_session_start_compact_captures_summary_as_candidate(project, tmp_path):
    engine = _engine(project)
    asyncio.run(engine.remember_decision("Use Postgres.", subject="database"))
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "summary", "summary": "We migrated auth to RS256 tokens."}) + "\n",
        encoding="utf-8",
    )
    result = handle_session_start(_payload(project, source="compact", transcript_path=str(transcript)))
    assert result is not None  # briefing still injected after compaction
    episodes = asyncio.run(_engine(project).bank.list(kind=MemoryKind.EPISODE))
    compact = [e for e in episodes if e.subject == "compact-summary"]
    assert len(compact) == 1
    assert "candidate" in compact[0].tags
    assert "RS256" in compact[0].text


# -- PreToolUse (strict mode) -----------------------------------------------------------


def test_pre_tool_default_mode_never_blocks(project):
    _seed_facts(project, ("The database listens on port 5432.", "db-port"))
    payload = _payload(project, tool_name="Grep", tool_input={"pattern": "database port"})
    assert handle_pre_tool(payload) is None


def test_pre_tool_strict_denies_when_memory_is_direct(project):
    (project / ".ragnite" / "config.toml").write_text("[invoke]\nstrict = true\n", encoding="utf-8")
    _seed_facts(
        project,
        ("The database listens on port 5432.", "db-listen"),
        ("The database port is 5432 in production.", "db-port-prod"),
        ("Postgres database uses port 5432.", "db-postgres"),
    )
    payload = _payload(project, tool_name="Grep", tool_input={"pattern": "database port"})
    result = handle_pre_tool(payload)
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "5432" in output["permissionDecisionReason"]
    # but an unrelated search passes even in strict mode
    other = _payload(project, tool_name="Grep", tool_input={"pattern": "websocket reconnect"})
    assert handle_pre_tool(other) is None


# -- PostToolUse: incremental re-index + cache invalidation ("FileChanged") -------------


def test_post_tool_edit_reindexes_file_and_invalidates_cache(project):
    app = project / "app.py"
    app.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    engine = _engine(project)
    asyncio.run(engine.index_repo(project))
    asyncio.run(engine.remember_fact("The database listens on port 5432.", "db-port"))
    asyncio.run(engine.recall("which port does the database use?"))  # warms the verdict cache
    assert asyncio.run(engine.cache.count()) >= 1

    app.write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n", encoding="utf-8")
    result = handle_post_tool(_payload(project, tool_name="Edit", tool_input={"file_path": str(app)}))
    assert result is None  # silent on purpose

    fresh = _engine(project)
    symbols = {
        r.metadata.get("symbol")
        for r in asyncio.run(fresh.bank.list(kind=MemoryKind.CODE))
        if r.metadata.get("symbol")
    }
    assert "beta" in symbols
    assert asyncio.run(fresh.cache.count()) == 0  # semantic cache invalidated
    stats = SessionState(project).data["stats"]
    assert stats["files_reindexed"] == 1
    assert stats["cache_invalidations"] >= 1


def test_post_tool_unchanged_edit_is_a_noop(project):
    app = project / "app.py"
    app.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    asyncio.run(_engine(project).index_repo(project))
    handle_post_tool(_payload(project, tool_name="Edit", tool_input={"file_path": str(app)}))
    assert SessionState(project).data["stats"]["files_reindexed"] == 0


# -- PostToolUse: episodic learning from Bash --------------------------------------------


def test_post_tool_bash_learns_candidate_episode_and_redacts(project):
    stdout = "2 failed, 5 passed in 0.31s token=sk-abcdefghijklmnopqrstu"
    result = handle_post_tool(
        _payload(
            project,
            tool_name="Bash",
            tool_input={"command": "uv run pytest -q"},
            tool_response={"stdout": stdout},
        )
    )
    assert result is not None
    assert "recorded candidate episode" in result["hookSpecificOutput"]["additionalContext"]

    episodes = asyncio.run(_engine(project).bank.list(kind=MemoryKind.EPISODE))
    assert len(episodes) == 1
    episode = episodes[0]
    assert "candidate" in episode.tags and "auto" in episode.tags
    assert "2 failed" in episode.text
    assert "sk-abcdefghijk" not in episode.text and "[REDACTED" in episode.text

    # repeat run supersedes instead of piling up
    handle_post_tool(
        _payload(
            project,
            tool_name="Bash",
            tool_input={"command": "uv run pytest -q"},
            tool_response={"stdout": "7 passed in 0.28s"},
        )
    )
    episodes = asyncio.run(_engine(project).bank.list(kind=MemoryKind.EPISODE))
    assert len(episodes) == 1
    assert "7 passed" in episodes[0].text


def test_post_tool_bash_failure_learned_and_boring_output_ignored(project):
    handle_post_tool(
        _payload(
            project,
            tool_name="Bash",
            tool_input={"command": "python app.py"},
            tool_response={"stdout": "Traceback (most recent call last)\nValueError: boom"},
        )
    )
    handle_post_tool(
        _payload(
            project,
            tool_name="Bash",
            tool_input={"command": "ls -la"},
            tool_response={"stdout": "total 8\n drwxr-xr-x"},
        )
    )
    episodes = asyncio.run(_engine(project).bank.list(kind=MemoryKind.EPISODE))
    assert len(episodes) == 1
    assert "ValueError" in episodes[0].text


# -- lifecycle + robustness -----------------------------------------------------------------


def test_invoke_toggle_persists(project):
    state = SessionState(project)
    state.set_active(False)
    assert SessionState(project).active is False
    state.set_active(True)
    assert SessionState(project).active is True


def test_run_hook_never_raises(project, monkeypatch):
    monkeypatch.chdir(project)
    assert run_hook("user-prompt", "this is not json{{{") is None
    assert run_hook("unknown-event", "{}") is None
    assert run_hook("user-prompt", "") is None


def test_run_hook_tolerates_bom_prefixed_stdin(project):
    _seed_facts(project, ("The database listens on port 5432.", "db-port"))
    payload = json.dumps(_payload(project, prompt="which port does the database use?"))
    output = run_hook("user-prompt", "﻿" + payload)
    assert output is not None
    assert "ragnite-context" in output


# -- bootstrap (/ragnite init) ------------------------------------------------------------


def test_bootstrap_init_seeds_inferred_memories(project):
    (project / "README.md").write_text(
        "# DemoProj\n\nA small demo service for parsing invoices.\n", encoding="utf-8"
    )
    (project / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demoproj"\nversion = "0.1.0"\n'
        '[project.scripts]\ndemoproj = "app:main"\n'
        '[dependency-groups]\ndev = ["pytest>=8"]\n',
        encoding="utf-8",
    )

    stats = asyncio.run(run_init(project))
    assert stats["code"]["files_indexed"] >= 1
    assert stats["doc_chunks"] >= 1
    assert stats["seeded_inferred"] >= 3
    assert len(stats["smoke"]) == 2
    assert all(
        s["mode"] in {"direct", "cautious", "ask_clarification", "search_more", "refuse_guess"}
        for s in stats["smoke"]
    )

    engine = _engine(project)
    facts = asyncio.run(engine.bank.list(kind=MemoryKind.FACT))
    inferred = [f for f in facts if "inferred" in f.tags]
    assert inferred and all(f.metadata.get("inferred") is True for f in inferred)
    assert all(f.authority <= 0.55 for f in inferred)  # never stored as definitive
    subjects = {f.subject for f in inferred}
    assert "project-brief" in subjects and "entry-points" in subjects

    # re-init replaces inferred records instead of duplicating
    before = len(inferred)
    asyncio.run(run_init(project))
    facts = asyncio.run(_engine(project).bank.list(kind=MemoryKind.FACT))
    assert len([f for f in facts if "inferred" in f.tags]) == before


def test_bootstrap_respects_ragniteignore_and_sensitive_files(project):
    (project / ".ragniteignore").write_text("private/\n", encoding="utf-8")
    private = project / "private"
    private.mkdir()
    (private / "secret_plan.py").write_text("def hidden():\n    pass\n", encoding="utf-8")
    (project / ".env").write_text("OPENAI_API_KEY=sk-abcdefghijklmnopqrstu\n", encoding="utf-8")
    (project / "ok.py").write_text("def visible():\n    pass\n", encoding="utf-8")

    asyncio.run(run_init(project))
    records = asyncio.run(_engine(project).bank.list(kind=MemoryKind.CODE))
    files = {r.metadata.get("file") for r in records}
    assert "ok.py" in files
    assert all(f and "private" not in f and ".env" not in f for f in files)
