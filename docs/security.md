# Security Model — what Ragnite stores, and what it refuses to

A live memory layer sees prompts, command output and file contents. The rules
below bound what can end up persisted in `.ragnite/`.

## Never stored

- **Sensitive files are never ingested or indexed**: `.env*`, private keys
  (`*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `*.p12`, `*.pfx`, `*.jks`),
  `credentials*`, `secrets*`, `.netrc`, `.npmrc`, `.pypirc`, cookie/ssh paths.
- **Ignored directories are never walked**: `.git`, `node_modules`, `dist`,
  `build`, `vendor`, `.venv`, `__pycache__`, `.claude`, `.ragnite`, etc.
- **`.ragniteignore`** (project root, one glob per line) excludes anything
  else — matched against the relative path, any parent directory, or the
  basename.

## Redacted before storage

Every string persisted from a live session — prompts used as semantic-cache
keys, Bash-derived episodes, compaction summaries, ingested doc text, and the
code snippets stored by Code Memory — passes through `redact()` first.
Patterns replaced with `[REDACTED]`:

- OpenAI/Anthropic-style keys (`sk-...`), AWS access key ids (`AKIA...`)
- GitHub (`ghp_/gho_/ghu_/ghs_/ghr_`, `github_pat_`), GitLab (`glpat-`),
  Slack (`xox?-`) tokens
- JWTs (`eyJ...x.y.z`), `Bearer <token>` headers
- passwords inside connection URLs (`postgres://user:****@host`)
- PEM private key blocks (entire block replaced)
- credential-ish assignments: `api_key= / secret: / password= / token=` values

Redaction is pattern-based, not perfect: a secret with no recognizable shape
in a string we store (e.g. inside a test-result line) can slip through. Treat
`.ragnite/` with the same sensitivity as your shell history.

## Operational boundaries

- `.ragnite/` is added to `.gitignore` at install time — runtime memory is
  never committed.
- Memory content is stored **unencrypted** on disk; with `RAGNITE_STORE=qdrant`
  it lives in your Qdrant instance — secure that like any datastore.
- With a remote embedding provider configured, recalled queries and indexed
  text are sent to that provider for embedding. Use `RAGNITE_EMBEDDER=local`
  (sentence-transformers) or `none` (BM25-only) for air-gapped projects.
- The MCP server executes local ingestion/indexing by design — register it
  only with clients you trust (same trust level as the hooks themselves).
- Wrong or sensitive memories can always be removed: `/ragnite forget <id>`,
  or `ragnite clear` / deleting `.ragnite/` for a full reset.

## Reporting

Vulnerabilities: see [SECURITY.md](../SECURITY.md) (GitHub private reporting).
