# Invoke Mode — event-driven live context injection

Invoke mode makes Ragnite a *living* memory layer: instead of the agent
pulling memory when it remembers to, the Claude Code event stream pushes the
right memory at the right moment. This is **not** token-by-token streaming —
it is **event-driven live context injection**: discrete session events
(session start, prompt submit, tool calls, compaction) trigger recall, learning
and re-indexing, and the results are injected as `additionalContext`.

## Lifecycle

```
ragnite claude install      one-time wiring (skill + MCP + hooks + config)
/ragnite init               heavy bootstrap (index code + docs, seed memories)
/ragnite invoke             active = true  -> injection begins
   ... normal work ...      events keep memory fresh and context flowing
/ragnite pause              active = false -> injection stops, memory kept
```

The active flag lives in `.ragnite/session.json`; every handler checks it
first, so paused mode costs one JSON read per event and nothing else.

## What happens on each event

| Session event | Ragnite reaction | Injected? |
|---|---|---|
| **SessionStart** (startup/resume/clear) | Build project briefing: brief, active decisions, constraints, memory counts | yes — `additionalContext` |
| **SessionStart** (compact) | Capture the compaction summary as a *candidate* episode, then re-inject the briefing (re-grounding after context loss) | yes |
| **UserPromptSubmit** | `recall(prompt)` → verdict + packed context as a `<ragnite-context mode=... confidence=...>` block; nothing injected on `refuse_guess` or low confidence | conditionally |
| **PreToolUse** (Grep/Glob) | Default: never blocks. `strict = true`: denies broad searches that memory answers `direct`, returning the context in the denial reason | strict only |
| **PostToolUse** (Edit/Write/...) | Incremental Code Memory re-index of the changed file + semantic cache invalidation | no (silent) |
| **PostToolUse** (Bash) | Learn *candidate* episodes from test results and failing commands (superseding repeats — one active episode per command) | tiny note when learned |

## The injected block

```
<ragnite-context mode="direct" confidence="0.84" cached="false">
suggestion: Strong consolidated memory. Answer directly from the provided context — do not re-derive...
- [decision|sim 0.81|3mo|adr/007.md] api-style: Services communicate over gRPC.
- [code|sim 0.74|2d|src/auth/jwt.py:41] src/auth/jwt.py::verify_token: ...
sources: adr/007.md, src/auth/jwt.py:41
</ragnite-context>
```

The mode is a contract the skill teaches Claude to obey: `direct` = answer
from context without re-reading; `ask_clarification` = memory conflicts, ask
the user; `search_more`/`refuse_guess` = nothing injected or investigate
normally.

## Learning discipline

Automatically learned knowledge is never promoted to definitive:

- Bash-derived episodes and compaction summaries carry the `candidate` +
  `auto` tags.
- Bootstrap inferences carry `inferred` (metadata + tag) and authority 0.5.
- Promotion is explicit: a human (or Claude, after confirming) stores the
  cleaned-up fact/decision via `remember`, superseding as needed.

## Cost profile

Each event spawns one short-lived Python process (hook) that loads the native
store from `.ragnite/` — millisecond-scale engine work; with a real embedding
provider, UserPromptSubmit adds one embedding API call (the verdict cache
absorbs repeats). Hooks time out independently (15–60 s budgets) and a hook
failure is logged to `.ragnite/hooks.log` and otherwise invisible — invoke
mode must never break a session.
