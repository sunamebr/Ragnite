# Hook Reference

How Ragnite's invoke mode maps onto real Claude Code hook events, and the
exact I/O contracts. Conceptual events people ask for (PostToolBatch,
FileChanged, PostCompact) do not exist in Claude Code — this table is the
honest mapping:

| Conceptual event | Actual Claude Code event | Ragnite handler |
|---|---|---|
| Session bootstrap briefing | `SessionStart` (all sources) | `session-start` |
| Post-compaction capture | `SessionStart` with `source: "compact"` | `session-start` |
| Prompt enrichment | `UserPromptSubmit` | `user-prompt` |
| Search redirection | `PreToolUse` matcher `Grep\|Glob` | `pre-tool` |
| Tool-batch learning | `PostToolUse` matcher `Bash\|Edit\|Write\|MultiEdit\|NotebookEdit` (per call) | `post-tool` |
| FileChanged re-index | `PostToolUse` on Edit/Write tools | `post-tool` |

Installed entries in `.claude/settings.local.json` (appended by
`ragnite claude install`, idempotent, existing hooks untouched):

```json
{
  "hooks": {
    "SessionStart": [
      {"hooks": [{"type": "command", "command": "\"<python>\" -m ragnite.cli claude hook session-start", "timeout": 60}]}
    ],
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "\"<python>\" -m ragnite.cli claude hook user-prompt", "timeout": 30}]}
    ],
    "PreToolUse": [
      {"matcher": "Grep|Glob",
       "hooks": [{"type": "command", "command": "\"<python>\" -m ragnite.cli claude hook pre-tool", "timeout": 15}]}
    ],
    "PostToolUse": [
      {"matcher": "Bash|Edit|Write|MultiEdit|NotebookEdit",
       "hooks": [{"type": "command", "command": "\"<python>\" -m ragnite.cli claude hook post-tool", "timeout": 60}]}
    ]
  }
}
```

## I/O contracts

Handlers read the hook payload as JSON on stdin and emit JSON on stdout
(exit 0 always). Empty stdout = no-op.

**SessionStart / UserPromptSubmit / PostToolUse** (context injection):

```json
{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "<ragnite-context ...>...</ragnite-context>"}}
```

**PreToolUse** (strict mode denial only):

```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Ragnite memory already answers this... <context>"}}
```

## Guarantees

- **Never breaks a session**: `run_hook` catches everything, logs the
  traceback to `.ragnite/hooks.log`, and exits 0 silently.
- **Inactive = near-zero cost**: handlers bail after one `session.json` read.
- **No secrets stored**: prompts (cache keys), episode texts and compaction
  summaries pass through redaction first (see [security.md](security.md)).
- **Default mode never blocks tools**: only `strict = true` lets PreToolUse
  deny — and only Grep/Glob, and only when memory answers `direct`.

## Testing hooks by hand

```bash
echo '{"cwd": ".", "prompt": "which port does the database use?"}' | \
  python -m ragnite.cli claude hook user-prompt
```

Returns the injection JSON when invoke mode is active and memory has an
answer; nothing otherwise.
