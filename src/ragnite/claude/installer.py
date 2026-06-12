"""``ragnite claude install`` — wire Ragnite into a Claude Code project.

Creates (merging, never clobbering):
- ``.claude/skills/ragnite/SKILL.md``   — the /ragnite slash skill
- ``.mcp.json``                         — ragnite MCP server (project scope)
- ``.claude/settings.local.json``       — hooks for invoke mode (appended)
- ``.ragnite/config.toml``              — invoke-mode knobs
- ``.ragnite/session.json``             — runtime state (inactive)

Hook commands use the absolute interpreter (``<python> -m ragnite.cli``) so
they work regardless of PATH/venv activation in the hook subshell.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ragnite.claude.session import SessionState

SKILL_MD = """---
name: ragnite
description: Ragnite memory engine controls. Use when the user types /ragnite or asks to initialize, invoke, or pause Ragnite project memory, recall project knowledge, remember facts/decisions/episodes, or check Ragnite status.
---

# Ragnite — confidence-aware project memory

Subcommands (first argument selects one):

| Command | Action |
|---|---|
| `/ragnite init` | Run `{python} -m ragnite.cli claude init` via Bash. Show its stats output to the user. Heavy bootstrap: indexes code + docs, seeds initial memories (inferences are tagged `inferred`), runs a smoke recall. |
| `/ragnite invoke` | Run `{python} -m ragnite.cli claude invoke` via Bash. It activates event-driven live context injection and prints a project briefing — read it and treat it as project memory. From now on, obey injected `<ragnite-context>` blocks. |
| `/ragnite pause` | Run `{python} -m ragnite.cli claude pause`. Stops context injection (memory is kept). |
| `/ragnite status` | Run `{python} -m ragnite.cli claude status`. Report active flag, stats, and memory counts. |
| `/ragnite recall <query>` | Call the MCP tool `recall` (server `ragnite`) with the query. Obey the returned `mode`. |
| `/ragnite remember <kind> <content>` | Call the MCP tool `remember` with kind = fact \\| decision \\| episode. Pick a short `subject` key from the content. |
| `/ragnite forget <id>` | Call the MCP tool `forget` with the memory id. |

## Behavior contract while invoke mode is active

- `<ragnite-context mode="direct">` blocks injected into prompts are consolidated
  project memory: answer from them, do **not** re-read or re-derive what they
  already state.
- `mode="cautious"` -> use the context but state caveats.
- `mode="ask_clarification"` -> memory conflicts; ask the user one targeted
  question, then store the winning entry with `remember` (decision, with
  `supersedes` when replacing).
- `mode="search_more"` / `refuse_guess` -> the context is insufficient;
  investigate normally, then `remember` what was expensive to figure out.
- After fixing a bug, taking a decision, or discovering a stable fact, store it
  via the `remember` MCP tool with a short subject key.
"""

CONFIG_TOML = """# Ragnite invoke-mode configuration (see docs/invoke-mode.md)
[invoke]
strict = false            # true: deny broad Grep/Glob when memory answers "direct"
budget_tokens = 1200      # max packed-context tokens injected per prompt
min_confidence = 0.25     # below this, nothing is injected
max_briefing_decisions = 8
learn_from_bash = true    # learn episodes from failing commands / test runs
"""

# event -> (matcher or None, ragnite hook subcommand, timeout seconds)
HOOK_SPECS: list[tuple[str, str | None, str, int]] = [
    ("SessionStart", None, "session-start", 60),
    ("UserPromptSubmit", None, "user-prompt", 30),
    ("PreToolUse", "Grep|Glob", "pre-tool", 15),
    ("PostToolUse", "Bash|Edit|Write|MultiEdit|NotebookEdit", "post-tool", 60),
]

_MARKER = "ragnite.cli claude hook"  # how we recognize our own hook entries


def hook_command(subcommand: str) -> str:
    return f'"{sys.executable}" -m ragnite.cli claude hook {subcommand}'


def merge_hooks(settings: dict) -> tuple[dict, bool]:
    """Append Ragnite hook entries to a Claude settings dict. Existing hooks,
    permissions, and unrelated keys are preserved verbatim. Idempotent."""
    changed = False
    hooks = settings.setdefault("hooks", {})
    for event, matcher, subcommand, timeout in HOOK_SPECS:
        entries = hooks.setdefault(event, [])
        already = any(
            _MARKER in hook.get("command", "") for entry in entries for hook in entry.get("hooks", [])
        )
        if already:
            continue
        new_entry: dict = {
            "hooks": [{"type": "command", "command": hook_command(subcommand), "timeout": timeout}]
        }
        if matcher:
            new_entry["matcher"] = matcher
        entries.append(new_entry)
        changed = True
    return settings, changed


def merge_mcp(existing: dict) -> tuple[dict, bool]:
    servers = existing.setdefault("mcpServers", {})
    if "ragnite" in servers:
        return existing, False
    servers["ragnite"] = {"command": sys.executable, "args": ["-m", "ragnite.cli", "mcp"]}
    return existing, True


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = path.with_suffix(path.suffix + ".ragnite-backup")
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def install_into(root: str | Path) -> list[str]:
    """Install everything into a project. Returns human-readable action log."""
    root = Path(root).resolve()
    actions: list[str] = []

    # 1. /ragnite skill
    skill_path = root / ".claude" / "skills" / "ragnite" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(SKILL_MD.replace("{python}", sys.executable), encoding="utf-8")
    actions.append(f"skill: {skill_path.relative_to(root)}")

    # 2. MCP server (project scope)
    mcp_path = root / ".mcp.json"
    mcp_data, mcp_changed = merge_mcp(_load_json(mcp_path))
    if mcp_changed:
        _write_json(mcp_path, mcp_data)
        actions.append("mcp: ragnite server added to .mcp.json")
    else:
        actions.append("mcp: already configured")

    # 3. invoke-mode config + session state
    config_path = root / ".ragnite" / "config.toml"
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(CONFIG_TOML, encoding="utf-8")
        actions.append("config: .ragnite/config.toml")
    state = SessionState(root)
    if state.data.get("installed_at") is None:
        state.data["installed_at"] = time.time()
    state.save()
    actions.append("state: .ragnite/session.json (inactive)")

    # 4. hooks (merge into settings.local.json — never overwrite existing entries)
    settings_path = root / ".claude" / "settings.local.json"
    settings, hooks_changed = merge_hooks(_load_json(settings_path))
    if hooks_changed:
        _write_json(settings_path, settings)
        actions.append(f"hooks: {len(HOOK_SPECS)} events wired in .claude/settings.local.json")
    else:
        actions.append("hooks: already installed")

    # 5. make sure runtime state never gets committed
    gitignore = root / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        # exact-line check: ".ragniteignore" in .gitignore must not mask this
        if not any(line.strip().rstrip("/") == ".ragnite" for line in lines):
            with gitignore.open("a", encoding="utf-8") as handle:
                handle.write("\n# Ragnite runtime data\n.ragnite/\n")
            actions.append("gitignore: added .ragnite/")
    return actions
