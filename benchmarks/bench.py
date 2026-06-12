"""Offline micro-benchmarks for Ragnite.

Run:  uv run python benchmarks/bench.py

No API keys, no network. Uses the deterministic FakeEmbedder, so the numbers
measure *engine overhead* (retrieval, scoring, packing, cache) — not provider
latency. With a real embedding provider every cold recall additionally pays
one embedding API round-trip; a cached recall still skips retrieval, scoring
and packing entirely, and an AnswerCache hit additionally skips the LLM call.
"""

from __future__ import annotations

import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ragnite.embed.fake import FakeEmbedder
from ragnite.memory.bank import MemoryBank
from ragnite.memory.code_index import CodeMemory
from ragnite.memory.engine import MemoryEngine
from ragnite.memory.semcache import SemanticCache
from ragnite.rag.engine import RagEngine
from ragnite.store.native import NativeVectorStore

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def _ms(samples: list[float]) -> str:
    return f"{statistics.mean(samples) * 1000:.2f} ms (p50 {statistics.median(samples) * 1000:.2f})"


async def bench_recall(n_facts: int = 300, runs: int = 25) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        engine = MemoryEngine(
            bank=MemoryBank(embedder=FakeEmbedder(), path=Path(tmp) / "bank"),
            cache=SemanticCache(embedder=FakeEmbedder(), path=Path(tmp) / "semcache"),
        )
        records = []
        from ragnite.memory.types import MemoryKind, MemoryRecord

        for i in range(n_facts):
            records.append(
                MemoryRecord(
                    kind=MemoryKind.FACT,
                    text=f"Service number {i} listens on port {7000 + i} and depends on service {i // 2}.",
                    subject=f"service-{i}",
                )
            )
        await engine.bank.add(records)

        cold: list[float] = []
        for i in range(runs):
            start = time.perf_counter()
            await engine.recall(f"which port does service number {i} use?", use_cache=False)
            cold.append(time.perf_counter() - start)

        query = "which port does service number 7 use?"
        warm_answer = await engine.recall(query)  # writes the cache entry
        cached: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            answer = await engine.recall(query)
            cached.append(time.perf_counter() - start)
            assert answer.cached
        return {
            "facts": n_facts,
            "cold": cold,
            "cached": cached,
            "speedup": statistics.mean(cold) / statistics.mean(cached),
            "context_tokens_reused": warm_answer.tokens,
        }


async def bench_code_index() -> dict:
    src = ROOT / "src" / "ragnite"
    with tempfile.TemporaryDirectory() as tmp:
        code = CodeMemory(MemoryBank(embedder=FakeEmbedder(), path=Path(tmp) / "bank"))
        start = time.perf_counter()
        stats = await code.index_repo(src)
        cold_s = time.perf_counter() - start
        start = time.perf_counter()
        again = await code.index_repo(src)
        incremental_s = time.perf_counter() - start
        return {
            "files": stats.files_indexed,
            "symbols": stats.symbols,
            "cold_s": cold_s,
            "incremental_s": incremental_s,
            "skipped": again.files_skipped,
        }


async def bench_retrieval_quality() -> dict:
    golden = [
        ("Why does Mars appear red?", "mars.md"),
        ("Which planet has a giant storm bigger than Earth?", "jupiter.md"),
        ("Why combine keyword and semantic search?", "rag.md"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rag = RagEngine(store=NativeVectorStore(Path(tmp) / "col"), embedder=FakeEmbedder())
        await rag.ingest_path(ROOT / "examples" / "data")
        hits = 0
        for query, expected in golden:
            results = await rag.search(query, top_k=3)
            if any(expected in (r.chunk.source or "") for r in results):
                hits += 1
        return {"queries": len(golden), "hit_at_3": hits / len(golden)}


async def main() -> None:
    recall = await bench_recall()
    index = await bench_code_index()
    quality = await bench_retrieval_quality()

    table = Table(title="Ragnite micro-benchmarks (offline, FakeEmbedder — engine overhead only)")
    table.add_column("benchmark")
    table.add_column("result", justify="right")
    table.add_row(f"cold recall over {recall['facts']} facts", _ms(recall["cold"]))
    table.add_row("cached recall (semantic verdict cache)", _ms(recall["cached"]))
    table.add_row("cache speedup", f"{recall['speedup']:.1f}x")
    table.add_row("packed context reused per hit", f"~{recall['context_tokens_reused']} tokens")
    table.add_row(
        f"code indexing ({index['files']} files / {index['symbols']} symbols)",
        f"{index['cold_s']:.2f} s",
    )
    table.add_row(f"incremental re-index ({index['skipped']} unchanged)", f"{index['incremental_s']:.2f} s")
    table.add_row(
        f"retrieval quality fixture (hit@3, {quality['queries']} queries)", f"{quality['hit_at_3']:.2f}"
    )
    console.print(table)
    console.print(
        "[dim]With a real provider, cold recall adds one embedding API call; cached recall "
        "still skips retrieval+scoring, and an AnswerCache hit also skips the LLM call.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())
