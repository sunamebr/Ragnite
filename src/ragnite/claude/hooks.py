"""Invoke-mode hook handlers — event-driven live context injection.

Each handler is a pure-ish function: Claude Code hook payload (dict) in,
hook JSON output (dict) or ``None`` out. The CLI entrypoint
(``ragnite claude hook <event>``) wires stdin/stdout around them and is
hard-wrapped to never crash or block a session: any failure logs to
``.ragnite/hooks.log`` and exits 0 silently.

Event mapping (Claude Code has no PostToolBatch/FileChanged/PostCompact —
see docs/hooks.md):

    SessionStart                       -> project briefing; on source=compact,
                                          also capture the compact summary as
                                          a candidate episode
    UserPromptSubmit                   -> recall + <ragnite-context> injection
    PreToolUse  (Grep|Glob)            -> strict mode only: deny broad searches
                                          that memory answers "direct"
    PostToolUse (Edit/Write/...)       -> incremental code re-index + cache
                                          invalidation ("FileChanged")
    PostToolUse (Bash)                 -> learn candidate episodes from test
                                          runs and failing commands
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

from ragnite.claude.redact import redact
from ragnite.claude.session import InvokeConfig, SessionState, find_project_root, load_invoke_config
from ragnite.config import build_memory_engine
from ragnite.memory.engine import MemoryEngine
from ragnite.memory.packer import estimate_tokens
from ragnite.memory.types import MemoryAnswer, MemoryKind

_MIN_PROMPT_CHARS = 12


def _engine(root: Path) -> MemoryEngine:
    from ragnite.claude.bootstrap import project_config

    return build_memory_engine(project_config(root))


def _root_for(payload: dict) -> Path:
    return find_project_root(payload.get("cwd"))


# -- briefing (SessionStart) ------------------------------------------------------


async def _briefing(engine: MemoryEngine, cfg: InvokeConfig) -> str | None:
    decisions = await engine.bank.list(kind=MemoryKind.DECISION)
    facts = await engine.bank.list(kind=MemoryKind.FACT)
    if not decisions and not facts:
        return None

    lines = [
        "RAGNITE PROJECT MEMORY — consolidated knowledge for this repository.",
        "Before re-reading files or re-deriving project knowledge, call the MCP tool "
        "`recall` (server: ragnite) and obey its `mode`.",
    ]
    brief = next((f for f in facts if f.subject == "project-brief"), None)
    if brief:
        lines.append(f"\nBrief{' (inferred)' if 'inferred' in brief.tags else ''}: {brief.text[:400]}")

    active_decisions = sorted(decisions, key=lambda r: r.updated_at, reverse=True)
    if active_decisions:
        lines.append("\nActive decisions:")
        for record in active_decisions[: cfg.max_briefing_decisions]:
            subject = f"{record.subject}: " if record.subject else ""
            lines.append(f"- {subject}{record.text[:220]}")

    constraints = [f for f in facts if "constraint" in f.tags]
    if constraints:
        lines.append("\nConstraints:")
        for record in constraints[:6]:
            lines.append(f"- {record.text[:200]}")

    stats = await engine.stats()
    counts = stats["active_by_kind"]
    lines.append(
        f"\nMemory: {counts['fact']} facts, {counts['decision']} decisions, "
        f"{counts['episode']} episodes, {counts['code']} code records."
    )

    text = "\n".join(lines)
    while estimate_tokens(text) > 1000 and len(lines) > 3:
        lines.pop(-2)
        text = "\n".join(lines)
    return text


def _compact_summary(transcript_path: str | None) -> str | None:
    """Best-effort extraction of the compaction summary from the transcript."""
    if not transcript_path:
        return None
    path = Path(transcript_path)
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-200:]):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") == "summary" and entry.get("summary"):
            return str(entry["summary"])
        if entry.get("isCompactSummary"):
            message = entry.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict)]
                joined = "\n".join(t for t in texts if t)
                if joined:
                    return joined
    return None


def handle_session_start(payload: dict, root: Path | None = None) -> dict | None:
    root = root or _root_for(payload)
    state = SessionState(root)
    if not state.active:
        return None
    cfg = load_invoke_config(root)
    engine = _engine(root)

    if payload.get("source") == "compact":
        summary = _compact_summary(payload.get("transcript_path"))
        if summary:
            asyncio.run(
                _remember_latest(
                    engine,
                    f"Session compaction summary (candidate — confirm before trusting): "
                    f"{redact(summary)[:2000]}",
                    subject="compact-summary",
                    tags=["compact", "candidate", "auto"],
                )
            )
            state.bump("episodes_learned")

    briefing = asyncio.run(_briefing(engine, cfg))
    if not briefing:
        return None
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": briefing}}


# -- prompt enrichment (UserPromptSubmit) -------------------------------------------


def _format_context(answer: MemoryAnswer) -> str:
    sources: list[str] = []
    for evidence in answer.evidence[:8]:
        src = evidence.record.source or evidence.record.subject
        if src and src not in sources:
            sources.append(src)
    source_line = f"\nsources: {', '.join(sources[:5])}" if sources else ""
    return (
        f'<ragnite-context mode="{answer.mode}" confidence="{answer.confidence:.2f}" '
        f'cached="{str(answer.cached).lower()}">\n'
        f"suggestion: {answer.suggestion}\n"
        f"{answer.context}{source_line}\n"
        f"</ragnite-context>"
    )


def handle_user_prompt(payload: dict, root: Path | None = None) -> dict | None:
    root = root or _root_for(payload)
    state = SessionState(root)
    if not state.active:
        return None
    prompt = (payload.get("prompt") or "").strip()
    if len(prompt) < _MIN_PROMPT_CHARS or prompt.startswith("/"):
        return None

    cfg = load_invoke_config(root)
    engine = _engine(root)
    # the prompt becomes a cache key downstream — never store secrets in it
    answer = asyncio.run(engine.recall(redact(prompt), budget_tokens=cfg.budget_tokens))
    if answer.mode == "refuse_guess" or not answer.context:
        return None
    if answer.confidence < cfg.min_confidence:
        return None

    state.bump("prompts_enriched")
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": _format_context(answer),
        }
    }


# -- search redirection (PreToolUse, strict mode only) ------------------------------


def _search_query(tool_name: str, tool_input: dict) -> str | None:
    pattern = tool_input.get("pattern") or ""
    words = re.sub(r"[^\w\s]", " ", str(pattern)).split()
    if not words:
        return None
    return " ".join(words[:12])


def handle_pre_tool(payload: dict, root: Path | None = None) -> dict | None:
    root = root or _root_for(payload)
    state = SessionState(root)
    if not state.active:
        return None
    cfg = load_invoke_config(root)
    tool_name = payload.get("tool_name", "")
    if not cfg.strict or tool_name not in {"Grep", "Glob"}:
        return None  # default mode never blocks — advisory only via UserPromptSubmit

    query = _search_query(tool_name, payload.get("tool_input") or {})
    if not query:
        return None
    engine = _engine(root)
    answer = asyncio.run(engine.recall(query, budget_tokens=600))
    if answer.mode != "direct":
        return None
    state.bump("searches_redirected")
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Ragnite memory already answers this with high confidence — use the context "
                "below (or the MCP `recall` tool) instead of a broad search:\n" + answer.context[:1500]
            ),
        }
    }


# -- learning + incremental re-index (PostToolUse) -----------------------------------

_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
_TEST_CMD = re.compile(r"\b(pytest|unittest|npm +test|yarn +test|go +test|cargo +test|jest|vitest|tox)\b")
_TEST_RESULT = re.compile(r"^.*\b(\d+ (?:passed|failed|errors?)|FAILED|PASSED|OK|✓|✗).*$", re.MULTILINE)
_ERROR_LINE = re.compile(
    r"^.*(Traceback \(most recent call last\)|[A-Za-z_.]*(?:Error|Exception):|error\[|fatal:|panic:).*$",
    re.MULTILINE,
)


def _response_text(tool_response) -> str:
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        parts = [str(tool_response.get(key, "")) for key in ("stdout", "stderr", "output", "content")]
        joined = "\n".join(p for p in parts if p)
        return joined or json.dumps(tool_response, ensure_ascii=False)[:4000]
    return str(tool_response or "")


async def _remember_latest(engine: MemoryEngine, text: str, subject: str, tags: list[str]) -> None:
    """Keep at most one active auto-record per subject (supersede on repeat)."""
    previous = None
    for record in await engine.bank.list(kind=MemoryKind.EPISODE):
        if record.subject == subject and "auto" in record.tags:
            previous = record
            break
    await engine.remember(
        text,
        kind=MemoryKind.EPISODE,
        subject=subject,
        tags=tags,
        supersedes=previous.id if previous else None,
    )


def _learn_from_bash(command: str, output: str) -> tuple[str, str] | None:
    """Return (episode_text, subject) when the command output is worth keeping."""
    is_test = bool(_TEST_CMD.search(command))
    short_cmd = command.strip().splitlines()[0][:80]
    subject = f"bash:{hashlib.sha1(short_cmd.encode()).hexdigest()[:8]}"
    if is_test:
        result = _TEST_RESULT.search(output)
        if result:
            return (f"Test run `{short_cmd}`: {result.group(0).strip()[:300]}", subject)
        return None
    matches = list(_ERROR_LINE.finditer(output))
    if matches:
        # prefer the actual exception line over the "Traceback ..." header
        best = next((m for m in matches if re.search(r"(?:Error|Exception)\b", m.group(0))), matches[0])
        return (f"Command failed `{short_cmd}`: {best.group(0).strip()[:300]}", subject)
    return None


def handle_post_tool(payload: dict, root: Path | None = None) -> dict | None:
    root = root or _root_for(payload)
    state = SessionState(root)
    if not state.active:
        return None
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    if tool_name in _EDIT_TOOLS:
        file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not file_path:
            return None
        engine = _engine(root)

        async def reindex() -> bool:
            changed = await engine.code.index_file(root, file_path)
            if changed and engine.cache is not None:
                await engine.cache.clear()
            return changed

        if asyncio.run(reindex()):
            state.bump("files_reindexed")
            state.bump("cache_invalidations")
        return None  # silent — Claude already knows it edited the file

    if tool_name == "Bash" and load_invoke_config(root).learn_from_bash:
        command = str(tool_input.get("command") or "")
        output = _response_text(payload.get("tool_response"))
        learned = _learn_from_bash(command, output)
        if not learned:
            return None
        text, subject = learned
        engine = _engine(root)
        asyncio.run(_remember_latest(engine, redact(text), subject, tags=["auto", "candidate", "bash"]))
        state.bump("episodes_learned")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"ragnite: recorded candidate episode — {redact(text)[:200]}",
            }
        }
    return None


# -- CLI dispatch ---------------------------------------------------------------------

HANDLERS = {
    "session-start": handle_session_start,
    "user-prompt": handle_user_prompt,
    "pre-tool": handle_pre_tool,
    "post-tool": handle_post_tool,
}


def run_hook(event: str, stdin_text: str) -> str | None:
    """stdin JSON -> handler -> stdout JSON (or None). Never raises."""
    root: Path | None = None
    try:
        stdin_text = stdin_text.lstrip(chr(0xFEFF)).strip()  # tolerate BOM-prefixed pipes (Windows)
        payload = json.loads(stdin_text) if stdin_text else {}
        root = _root_for(payload)
        handler = HANDLERS.get(event)
        if handler is None:
            return None
        result = handler(payload, root)
        return json.dumps(result, ensure_ascii=False) if result else None
    except Exception:  # noqa: BLE001 — a hook must never break the user's session
        try:
            log = (root or Path.cwd()) / ".ragnite" / "hooks.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event}\n")
                handle.write(traceback.format_exc() + "\n")
        except OSError:
            pass
        return None
