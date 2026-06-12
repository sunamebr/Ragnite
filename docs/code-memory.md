# Code Memory

`index_repo(path)` turns a repository into queryable, confidence-scored
memory so "where is X handled?" is one recall instead of a directory crawl.

## What gets extracted

Per indexed file, one **file record**:

```
file src/auth/jwt.py (python, 6 symbols)
imports: jose, fastapi, app.config
symbols: verify_token, TokenService, TokenService.refresh, ...
```

…plus one **symbol record** per definition:

```
function verify_token [GET /verify] — src/auth/jwt.py:41
def verify_token(token: str) -> Claims:
Validate an RS256-signed JWT and return its claims.
```

| Field | Where it lands |
|---|---|
| file, line, language, content hash | `metadata` |
| symbol name + type (function/class/method) | `metadata`, `subject` = `path::symbol` |
| signature + docstring head | record text (what gets embedded/BM25-indexed) |
| HTTP route from decorators (`@app.get("/x")` style) | `metadata["route"]`, tag `endpoint` |
| test files (`test_*`, `*_test.py`, `*.test.*`, `*.spec.*`) | tag `test` |
| module imports | file record `metadata["imports"]` → `CodeMemory.graph()` |

## Parsing strategy per language

- **Python** — full `ast` parse: top-level functions, classes, methods
  (`Class.method`), docstrings, imports, decorator routes. Syntax errors
  degrade to zero symbols for that file (the file record still indexes).
- **JS/TS/Go/Rust/Java/Kotlin/Ruby/PHP/C#/C/C++/Swift/Scala/Lua/SQL** —
  definition-boundary regex (`function/fn/func/class/interface/struct/...`
  plus `const X = (` arrow functions) and import/require/use extraction.
  Shallower than AST, deliberately: structure + names + location is what
  recall needs.

## Incremental semantics

- Files are skipped when their content hash (sha256/16) matches the indexed
  one — re-running `index_repo` on an unchanged tree is a no-op.
- A changed file has its old records deleted before re-indexing (no dupes).
- Files that disappeared are evicted (`files_removed`).
- Directories in the standard ignore set (`.git`, `node_modules`, `.venv`,
  `dist`, …) and files > 512 KB are never indexed.

Measured on Ragnite's own source (47 files / 298 symbols, offline embedder):
cold index ≈ 0.14 s, incremental re-index ≈ 0.01 s — cheap enough to run at
every session start. Real embedding providers add one batched embedding call
for changed files only.

## Freshness

`code` records use a 21-day confidence half-life — stale code knowledge decays
fast, and re-indexing refreshes timestamps only for files that actually
changed. The right cadence: `index_repo` at session start (the agent-loop
contract), which keeps recency ≈ 1.0 for everything that matters.

## Limits (v0.2)

- Symbol granularity, not call graphs: `graph()` gives file→imports edges, not
  function-level call relations.
- Regex languages don't capture methods inside classes or docstrings.
- No cross-file type resolution — this is retrieval memory, not an LSP.
