# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's
private vulnerability reporting on this repository ("Security" tab → "Report a
vulnerability"). You should receive an acknowledgement within 72 hours.

## Scope notes for operators

- The HTTP API ships **without** authentication by default; set
  `RAGNITE_API_KEY` (bearer token) before exposing it beyond localhost, and
  terminate TLS in front of it.
- Ingested documents and the embedding cache are stored unencrypted in
  `RAGNITE_DATA_DIR`. Treat that directory like the data it contains.
- The MCP server executes ingestion of arbitrary local paths by design; only
  register it with MCP clients you trust.
- Never put API keys inside documents you ingest — chunks are stored, logged
  in eval reports, and may be sent to LLM providers as context.
