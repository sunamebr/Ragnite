"""``ragnite claude init`` — heavy project bootstrap.

Indexes code into Code Memory, ingests README/docs/configs into document RAG,
seeds initial memories, and runs a smoke recall. Seeded knowledge derived by
inspection (language mix, entry points, test framework, README brief) is never
stored as definitive: records carry ``metadata.inferred = true``, the
``inferred`` tag, and reduced authority. Re-running init replaces previous
inferred records instead of duplicating them.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ragnite.claude.redact import is_ignored, is_sensitive_path, load_ragniteignore, redact
from ragnite.config import RagniteConfig, build_engine, build_memory_engine
from ragnite.ingest.loaders import load_path
from ragnite.memory.engine import MemoryEngine
from ragnite.memory.types import MemoryKind, MemoryRecord

_DOC_GLOBS = ["README*", "*.md", "docs/**/*.md", "docs/**/*.rst", "docs/**/*.txt"]
_CONFIG_FILES = [
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "composer.json",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    "Makefile",
]
_INFERRED_AUTHORITY = 0.5


def project_config(root: Path) -> RagniteConfig:
    cfg = RagniteConfig.from_env()
    cfg.data_dir = root / ".ragnite"
    return cfg


async def _replace_inferred(engine: MemoryEngine, subject: str) -> None:
    for record in await engine.bank.list():
        if record.subject == subject and "inferred" in record.tags:
            await engine.bank.delete([record.id])


async def _seed_inferred(engine: MemoryEngine, text: str, subject: str, source: str | None) -> None:
    await _replace_inferred(engine, subject)
    record = MemoryRecord(
        kind=MemoryKind.FACT,
        text=redact(text),
        subject=subject,
        tags=["inferred"],
        source=source,
        authority=_INFERRED_AUTHORITY,
        metadata={"inferred": True},
    )
    await engine.bank.add([record])


def _readme_brief(root: Path) -> tuple[str, str] | None:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = root / name
        if readme.exists():
            text = readme.read_text(encoding="utf-8-sig", errors="replace")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith(("![", "[!"))]
            brief = " ".join(lines[:8])[:500]
            if brief:
                return brief, name
    return None


def _entry_points(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            scripts = data.get("project", {}).get("scripts", {})
            if scripts:
                listed = ", ".join(f"{k} -> {v}" for k, v in list(scripts.items())[:8])
                return f"Console entry points (pyproject.toml): {listed}"
        except tomllib.TOMLDecodeError:
            pass
    package = root / "package.json"
    if package.exists():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8-sig", errors="replace")).get("scripts", {})
            if scripts:
                listed = ", ".join(f"{k}: {v}" for k, v in list(scripts.items())[:8])
                return f"npm scripts (package.json): {listed}"
        except json.JSONDecodeError:
            pass
    return None


def _test_framework(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        raw = pyproject.read_text(encoding="utf-8-sig", errors="replace")
        if "pytest" in raw:
            return "Tests appear to use pytest (referenced in pyproject.toml)."
    package = root / "package.json"
    if package.exists():
        raw = package.read_text(encoding="utf-8-sig", errors="replace")
        for framework in ("vitest", "jest", "mocha", "playwright"):
            if framework in raw:
                return f"Tests appear to use {framework} (referenced in package.json)."
    return None


async def _language_mix(engine: MemoryEngine) -> str | None:
    counts: dict[str, int] = {}
    for record in await engine.bank.list(kind=MemoryKind.CODE):
        if record.metadata.get("symbol_type") == "file":
            lang = record.metadata.get("language", "unknown")
            counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return None
    mix = ", ".join(f"{n} {lang}" for lang, n in sorted(counts.items(), key=lambda i: -i[1]))
    return f"Code base composition (by indexed files): {mix}."


def _doc_files(root: Path, ignore: list[str]) -> list[Path]:
    seen: set[Path] = set()
    for pattern in _DOC_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                seen.add(path)
    for name in _CONFIG_FILES:
        path = root / name
        if path.is_file():
            seen.add(path)
    files: list[Path] = []
    for path in sorted(seen):
        rel = path.relative_to(root).as_posix()
        if is_sensitive_path(path) or is_ignored(rel, ignore):
            continue
        files.append(path)
    return files


async def run_init(root: str | Path) -> dict[str, Any]:
    """Bootstrap a project. Returns a stats dict for display."""
    root = Path(root).resolve()
    cfg = project_config(root)
    memory = build_memory_engine(cfg)
    rag = build_engine(cfg)
    ignore = load_ragniteignore(root)

    # 1. Code Memory (incremental, .ragniteignore-aware)
    code_stats = await memory.code.index_repo(root, ignore=ignore)

    # 2. Documents: README / docs / configs into the RAG collection (redacted)
    doc_chunks = 0
    doc_files = _doc_files(root, ignore)
    for path in doc_files:
        docs = load_path(path)
        for doc in docs:
            doc.text = redact(doc.text)
            doc.source = path.relative_to(root).as_posix()
        if docs:
            stats = await rag.ingest_documents(docs)
            doc_chunks += stats.chunks

    # 3. Seed inferred memories (never definitive: inferred=true, low authority)
    seeded = 0
    brief = _readme_brief(root)
    if brief:
        await _seed_inferred(memory, f"Project brief (from README): {brief[0]}", "project-brief", brief[1])
        seeded += 1
    mix = await _language_mix(memory)
    if mix:
        await _seed_inferred(memory, mix, "project-structure", None)
        seeded += 1
    entry = _entry_points(root)
    if entry:
        await _seed_inferred(memory, entry, "entry-points", None)
        seeded += 1
    tests = _test_framework(root)
    if tests:
        await _seed_inferred(memory, tests, "test-framework", None)
        seeded += 1

    # 4. A real (non-inferred) episode marking the bootstrap
    await memory.remember(
        f"Ragnite initialized: indexed {code_stats.files_indexed} code files "
        f"({code_stats.symbols} symbols), {doc_chunks} doc chunks from {len(doc_files)} files.",
        kind=MemoryKind.EPISODE,
        subject="ragnite-init",
        tags=["init"],
    )

    # 5. Smoke recall — prove the loop works before declaring success
    smoke = []
    for query in ("What is this project about?", "Where are the main entry points defined?"):
        answer = await memory.recall(query, use_cache=False)
        smoke.append({"query": query, "mode": answer.mode, "confidence": answer.confidence})

    return {
        "root": str(root),
        "code": code_stats.model_dump(),
        "doc_files": len(doc_files),
        "doc_chunks": doc_chunks,
        "seeded_inferred": seeded,
        "memory": await memory.stats(),
        "smoke": smoke,
    }
