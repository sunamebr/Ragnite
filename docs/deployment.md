# Deploying Ragnite

## Single node (simplest)

```bash
pip install "ragnite[server,anthropic]"
export ANTHROPIC_API_KEY=... VOYAGE_API_KEY=...
export RAGNITE_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
ragnite ingest /srv/knowledge
ragnite serve --host 0.0.0.0 --port 8000
```

Native store + embedding cache live in `RAGNITE_DATA_DIR` (default `.ragnite`).
Back that directory up; it is the entire index state.

## Docker Compose (API + Qdrant)

```bash
cd docker
ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... RAGNITE_API_KEY=... docker compose up --build
```

- `ragnite` service: HTTP API on :8000, data volume at `/data`.
- `qdrant` service: vector database with its own volume.

## Production checklist

- [ ] `RAGNITE_API_KEY` set; TLS terminated by a reverse proxy (Caddy/nginx/ALB).
- [ ] `RAGNITE_STORE=qdrant` for datasets beyond a single node's RAM, or for
      replication/HA. Native store is fine (and faster to operate) below that.
- [ ] Persistent volumes for `/data` and Qdrant storage.
- [ ] Liveness: `GET /healthz`. Readiness: `GET /v1/stats` (touches the store).
- [ ] Rate-limit at the proxy; the API itself is stateless and scales
      horizontally when backed by Qdrant.
- [ ] Run `ragnite eval` against a golden dataset in CI before promoting a new
      chunking/embedding configuration.

## Scaling notes

| Corpus size | Recommended setup |
|---|---|
| < 100k chunks | Native store, single node, no extra infra |
| 100k – 10M chunks | Qdrant (single instance, snapshots on) |
| > 10M chunks / HA | Qdrant cluster; shard collections per tenant; multiple stateless API replicas |

Embedding throughput is the usual ingest bottleneck — the SQLite cache makes
re-runs free, and `EmbeddingProvider.batch_size` controls request sizing.

## Kubernetes (sketch)

Run the container with env config, mount a PVC at `/data` (or use Qdrant and
keep API pods stateless), expose `/healthz` as liveness/readiness probes, and
keep provider keys in a `Secret`. A Helm chart is on the roadmap.
