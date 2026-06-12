# Ragnite × Claude Code

Turn Ragnite into a **live memory layer** for Claude Code: install once, run
`/ragnite init` for the heavy bootstrap, `/ragnite invoke` to activate
**event-driven live context injection**, then just work — Ragnite injects
consolidated memory before prompts, learns from tool calls, re-indexes edited
files, and survives compactions.

## Install

```bash
pip install "ragnite[mcp]"
cd your-project
ragnite claude install
```

This creates / merges (never clobbering existing config):

| Artifact | Purpose |
|---|---|
| `.claude/skills/ragnite/SKILL.md` | the `/ragnite` slash skill |
| `.mcp.json` | `ragnite` MCP server (recall/remember/index tools) |
| `.claude/settings.local.json` | 4 hooks (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse) — appended next to your existing hooks |
| `.ragnite/config.toml` | invoke-mode knobs (strict, budgets, thresholds) |
| `.ragnite/session.json` | runtime state (starts **inactive**) |

Hook and MCP commands use the absolute interpreter
(`"<python>" -m ragnite.cli ...`), so they work regardless of PATH or venv
activation. Restart the Claude Code session after installing so hooks, skill
and MCP server load.

## Bootstrap: `/ragnite init`

Heavy one-time pass (incremental afterwards):

1. Detects the project root (`.ragnite`/`.git` upward walk).
2. Indexes the code base into **Code Memory** (`.ragniteignore`-aware).
3. Ingests README / docs / configs into the document collection (redacted).
4. Seeds initial memories — language mix, entry points, test framework,
   README brief. **Inferences are never stored as definitive**: they carry
   `metadata.inferred = true`, the `inferred` tag, and authority 0.5. Confirm
   or correct them (`/ragnite remember`, `/ragnite forget`).
5. Runs a smoke recall and prints stats.

## Activate: `/ragnite invoke`

Sets `.ragnite/session.json` → `active: true`, validates that the MCP server
and hooks are installed (warns if not), prints the project briefing into the
conversation, and instructs Claude to call `recall` before re-analyzing the
repo. From then on, every hook fires live context injection (see
[invoke-mode.md](invoke-mode.md)).

`/ragnite pause` deactivates injection (memory is kept).
`/ragnite status` shows the active flag, counters and memory stats.

## Day-to-day commands

| Command | Effect |
|---|---|
| `/ragnite recall <query>` | MCP `recall` — verdict + packed context |
| `/ragnite remember <kind> <content>` | store fact / decision / episode |
| `/ragnite forget <id>` | delete a wrong memory |
| `ragnite claude status` | same as `/ragnite status`, from any shell |

## Configuration (`.ragnite/config.toml`)

```toml
[invoke]
strict = false            # true: deny broad Grep/Glob answered "direct" by memory
budget_tokens = 1200      # packed-context budget per prompt injection
min_confidence = 0.25     # below this nothing is injected (no noise)
max_briefing_decisions = 8
learn_from_bash = true    # learn episodes from failing commands / test runs
```

## Troubleshooting

- **Nothing is injected** → `/ragnite status`: check `active: true` and
  `install_problems: []`; hooks require a session restart after install.
- **Hook errors** → `.ragnite/hooks.log` (handlers never crash the session;
  they log and stay silent).
- **Too chatty / too quiet** → tune `min_confidence` and `budget_tokens`.
- **Stale answers after manual edits outside Claude** → run `/ragnite init`
  (incremental — unchanged files skip).
