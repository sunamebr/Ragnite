"""MCP server (optional extra ``ragnite[mcp]``).

Exposes Ragnite's confidence-aware memory + document RAG to MCP clients
(Claude Code, Claude Desktop, any MCP host) over stdio.

The memory tools are the point: an agent calls ``recall`` once and receives a
packed minimal context plus an explicit answer mode (direct / cautious /
ask_clarification / search_more / refuse_guess) — instead of re-analyzing the
project and burning tokens every session.

Register with Claude Code:
    claude mcp add ragnite -- ragnite mcp
"""

from __future__ import annotations

import json

from ragnite.config import RagniteConfig, build_engine, build_memory_engine
from ragnite.errors import MissingDependencyError


def create_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingDependencyError("mcp", "mcp") from exc

    from ragnite.memory.types import MemoryKind

    cfg = RagniteConfig.from_env()
    engine = build_engine(cfg)
    memory = build_memory_engine(cfg)
    mcp = FastMCP("ragnite")

    # -- confidence-aware memory ------------------------------------------------

    @mcp.tool()
    async def recall(query: str, budget_tokens: int = 2000, kinds: str = "") -> str:
        """Recall consolidated project memory BEFORE re-reading files or re-analyzing.
        Returns a confidence verdict: 'mode' tells you how to proceed (direct /
        cautious / ask_clarification / search_more / refuse_guess), 'context' is the
        minimal evidence pack, 'suggestion' is the instruction to follow.
        Optional kinds filter: comma-separated among fact,decision,episode,code."""
        kind_list = [MemoryKind(k.strip()) for k in kinds.split(",") if k.strip()] if kinds else None
        answer = await memory.recall(query, kinds=kind_list, budget_tokens=budget_tokens)
        return json.dumps(
            {
                "mode": answer.mode,
                "confidence": answer.confidence,
                "suggestion": answer.suggestion,
                "context": answer.context,
                "tokens": answer.tokens,
                "cached": answer.cached,
                "signals": answer.signals.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    async def remember(text: str, kind: str = "fact", subject: str = "", source: str = "") -> str:
        """Store consolidated knowledge so it never has to be re-derived.
        kind: 'fact' (stable project/domain truth), 'decision' (architectural or
        strategic decision taken), 'episode' (bug fixed, failed attempt, progress).
        subject: short topic key (e.g. 'db-port', 'auth-strategy') — enables
        conflict detection when entries disagree."""
        record = await memory.remember(text, kind=kind, subject=subject or None, source=source or None)
        return f"Remembered {record.kind.value} {record.id}."

    @mcp.tool()
    async def remember_decision(text: str, subject: str = "", supersedes: str = "", source: str = "") -> str:
        """Store an architectural/strategic decision. Pass supersedes=<memory id>
        when this decision replaces an earlier one — the old entry is retired."""
        record = await memory.remember_decision(
            text, subject=subject or None, supersedes=supersedes or None, source=source or None
        )
        return f"Remembered decision {record.id}" + (f" (supersedes {supersedes})." if supersedes else ".")

    @mcp.tool()
    async def index_repo(path: str) -> str:
        """Index a repository into Code Memory (files, symbols, imports, endpoints,
        tests). Incremental: unchanged files are skipped. Run once per session start
        instead of re-reading the codebase."""
        stats = await memory.index_repo(path)
        return (
            f"Indexed {stats.files_indexed} file(s) / {stats.symbols} symbol(s); "
            f"skipped {stats.files_skipped} unchanged; removed {stats.files_removed} stale."
        )

    @mcp.tool()
    async def forget(memory_id: str) -> str:
        """Delete a memory record that turned out to be wrong."""
        return "Forgotten." if await memory.forget(memory_id) else "No such memory id."

    # -- document RAG ----------------------------------------------------------------

    @mcp.tool()
    async def search(query: str, top_k: int = 6) -> str:
        """Hybrid search (semantic + keyword) over the ingested document base."""
        results = await engine.search(query, top_k=top_k)
        if not results:
            return "No results."
        return json.dumps(
            [
                {
                    "score": round(r.score, 4),
                    "source": r.chunk.source,
                    "text": r.chunk.text[:1200],
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    async def ask(question: str, top_k: int = 6) -> str:
        """Answer a question from the ingested documents with source citations."""
        answer = await engine.ask(question, top_k=top_k)
        sources = "\n".join(f"[{c.marker}] {c.source or c.doc_id}" for c in answer.citations)
        return f"{answer.text}\n\nSources:\n{sources}" if sources else answer.text

    @mcp.tool()
    async def ingest_text(text: str, source: str = "inline") -> str:
        """Add a piece of text to the document base."""
        stats = await engine.ingest_text(text, source=source)
        return f"Ingested {stats.chunks} chunk(s) from {stats.documents} document(s)."

    @mcp.tool()
    async def ingest_path(path: str) -> str:
        """Ingest a local file or directory (md, txt, code, html, json, pdf, docx)."""
        stats = await engine.ingest_path(path)
        return f"Ingested {stats.chunks} chunk(s) from {stats.documents} document(s)."

    @mcp.tool()
    async def stats() -> str:
        """Index + memory statistics."""
        info = await engine.stats()
        info["memory"] = await memory.stats()
        return json.dumps(info, ensure_ascii=False, indent=2)

    return mcp


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
