"""MCP server (optional extra ``ragnite[mcp]``).

Exposes Ragnite to MCP clients (Claude Code, Claude Desktop, any MCP host)
over stdio. Tools: search, ask, ingest, plus persistent vector memory
(remember / recall) so agents keep knowledge across sessions.

Register with Claude Code:
    claude mcp add ragnite -- ragnite mcp
"""

from __future__ import annotations

import json

from ragnite.config import RagniteConfig, build_engine, build_memory
from ragnite.errors import MissingDependencyError


def create_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingDependencyError("mcp", "mcp") from exc

    cfg = RagniteConfig.from_env()
    engine = build_engine(cfg)
    memory = build_memory(cfg)
    mcp = FastMCP("ragnite")

    @mcp.tool()
    async def search(query: str, top_k: int = 6) -> str:
        """Hybrid search (semantic + keyword) over the indexed knowledge base.
        Returns the most relevant passages with their sources and scores."""
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
        """Answer a question from the knowledge base with source citations."""
        answer = await engine.ask(question, top_k=top_k)
        sources = "\n".join(f"[{c.marker}] {c.source or c.doc_id}" for c in answer.citations)
        return f"{answer.text}\n\nSources:\n{sources}" if sources else answer.text

    @mcp.tool()
    async def ingest_text(text: str, source: str = "inline") -> str:
        """Add a piece of text to the knowledge base."""
        stats = await engine.ingest_text(text, source=source)
        return f"Ingested {stats.chunks} chunk(s) from {stats.documents} document(s)."

    @mcp.tool()
    async def ingest_path(path: str) -> str:
        """Index a local file or directory (md, txt, code, html, json, pdf, docx)."""
        stats = await engine.ingest_path(path)
        return f"Ingested {stats.chunks} chunk(s) from {stats.documents} document(s)."

    @mcp.tool()
    async def remember(fact: str) -> str:
        """Store a fact in persistent vector memory for future sessions."""
        memory_id = await memory.remember(fact)
        return f"Remembered ({memory_id})."

    @mcp.tool()
    async def recall(query: str, top_k: int = 5) -> str:
        """Recall facts from persistent vector memory by semantic similarity."""
        results = await memory.recall(query, k=top_k)
        if not results:
            return "Nothing relevant in memory."
        return "\n".join(f"- {r.chunk.text}" for r in results)

    @mcp.tool()
    async def stats() -> str:
        """Index statistics: chunk count, store, embedder, models."""
        info = await engine.stats()
        info["memories"] = await memory.count()
        return json.dumps(info, ensure_ascii=False, indent=2)

    return mcp


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
