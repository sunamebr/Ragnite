# Contributing to Ragnite

Thanks for helping! The bar: every change ships with tests and passes CI.

## Setup

```bash
git clone https://github.com/sunamebr/Ragnite
cd Ragnite
uv sync --group dev          # creates .venv with dev tools
uv run pytest                # full suite runs offline, no API keys needed
uv run ruff check . && uv run ruff format .
```

## Ground rules

- **Offline-first tests.** The test suite must never hit a network. Use
  `FakeEmbedder` and the `FakeChat` test double; new providers get thin
  adapters + (optionally) integration tests gated behind env vars.
- **Optional deps stay optional.** Anything beyond `pydantic/numpy/httpx/typer`
  goes behind an extra and a lazy import (`MissingDependencyError` on use).
- **Async-first.** All I/O paths are `async`; CPU-bound work goes through
  `asyncio.to_thread`.
- **Small PRs.** One feature or fix per PR, with a short "what / why".

## Adding a provider

1. Implement the interface (`EmbeddingProvider`, `VectorStore`, `ChatModel`, or `Reranker`).
2. Lazy-import the SDK; raise `MissingDependencyError(pkg, extra)` when missing.
3. Wire it into `ragnite/config.py` factories + the env table in the README.
4. Add unit tests with mocked HTTP (e.g. `httpx.MockTransport`) where practical.

## Releases

Maintainers tag `vX.Y.Z`; the release workflow builds and publishes to PyPI
via trusted publishing.
