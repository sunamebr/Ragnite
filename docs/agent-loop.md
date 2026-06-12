# The Agent Loop

How a coding agent (Claude Code, or any MCP host) is supposed to use Ragnite.
The contract is four habits; the MCP tool descriptions already teach them, but
a system-prompt reinforcement makes agents follow them consistently.

## The four habits

1. **Session start:** `index_repo(".")` — incremental, unchanged files are
   hash-skipped, so this costs near-nothing after the first run.
2. **Before re-reading or re-deriving anything:** `recall(question)`.
3. **Obey the returned `mode`:**
   - `direct` → answer from `context`; do not re-open files.
   - `cautious` → answer with caveats, attribute claims to memory entries.
   - `ask_clarification` → memory conflicts; ask the user, then record the
     winning entry with `remember_decision(supersedes=...)`.
   - `search_more` → the memory layer itself is telling you it doesn't know
     enough: go read code/docs, then `remember` what you learned.
   - `refuse_guess` → say you don't know.
4. **After anything expensive to figure out:** `remember(...)` with a good
   `subject`; decisions that replace older ones pass `supersedes`.

## System prompt snippet (copy-paste)

```
You have Ragnite memory tools. Rules:
- At session start, call index_repo on the project root.
- Before re-reading files or re-deriving project knowledge, call recall and
  obey its "mode". When mode is "direct", answer from the provided context
  without re-analyzing the repository.
- When you fix a bug, take a decision, or discover a stable fact, store it
  with remember (kind: episode / decision / fact) and a short subject key.
- When a decision replaces an earlier one, use remember_decision with
  supersedes=<old id> instead of leaving both active.
```

## A real session, played out

```
user:  "Where do we validate JWTs, and what alg do we use?"

agent: recall("where are JWTs validated and which algorithm")
  -> mode: direct (0.83)
     context:
     - [code|sim 0.78|2d|src/auth/jwt.py:41] src/auth/jwt.py::verify_token: function verify_token ...
     - [decision|sim 0.71|4mo|adr/009.md] jwt-alg: We sign JWTs with RS256; HS256 is forbidden.

agent: answers directly. Zero file reads. (~40 input tokens of context instead
       of re-reading auth/* — typically thousands.)

user:  "Switch token TTL to 15 minutes."

agent: edits code, runs tests, then:
       remember_decision("Access-token TTL is 15 minutes.", subject="jwt-ttl",
                          supersedes="mem_a1b2...")   # the old 60-min decision
       remember("Lowering TTL broke the refresh test until we adjusted the
                 clock skew allowance.", kind="episode", subject="jwt-ttl")
```

The flywheel: each session writes back what it learned, so the next session's
`recall` answers `direct` more often — and `direct` means *tokens not spent*.

## Where the savings actually come from

| Without Ragnite | With Ragnite |
|---|---|
| Re-read `auth/*` every session (often thousands of tokens of file content) | One `recall` returns a ~100–2000-token packed context |
| Re-derive past decisions from code archaeology | Decision Memory answers in one entry, with lineage |
| Repeat a previously-failed approach | Episodic Memory surfaces "we tried X, it broke Y" |
| Same question re-answered every time | Verdict cache returns instantly; AnswerCache (opt-in) even skips the LLM |

The honest accounting: the **verdict cache** saves retrieval/scoring and lets
the host reuse the packed context; only the **AnswerCache** (document RAG,
opt-in) makes a repeat question literally zero LLM tokens. See
[semantic-cache.md](semantic-cache.md).
