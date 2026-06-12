"""Ragnite CLI."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from ragnite.config import RagniteConfig, build_engine

app = typer.Typer(
    name="ragnite",
    help="Production-grade RAG: ingest, search, ask, serve, evaluate.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


def _engine():
    return build_engine(RagniteConfig.from_env())


@app.command()
def ingest(
    paths: list[str] = typer.Argument(..., help="Files or directories to index."),
    recursive: bool = typer.Option(True, help="Recurse into directories."),
) -> None:
    """Index files or directories into the knowledge base."""

    async def run() -> None:
        engine = _engine()
        for path in paths:
            stats = await engine.ingest_path(path, recursive=recursive)
            console.print(
                f"[green]+[/green] {path}: {stats.chunks} chunks / {stats.documents} docs"
                f"{' (contextualized)' if stats.contextualized else ''}"
            )

    asyncio.run(run())


@app.command()
def query(
    text: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(6, help="Number of results."),
) -> None:
    """Hybrid search — show retrieved chunks without calling an LLM."""

    async def run() -> None:
        results = await _engine().search(text, top_k=top_k)
        if not results:
            console.print("[yellow]no results[/yellow]")
            return
        table = Table(show_lines=True)
        table.add_column("#", width=3)
        table.add_column("score", width=8)
        table.add_column("source", max_width=36, overflow="fold")
        table.add_column("text", overflow="fold")
        for i, scored in enumerate(results, start=1):
            table.add_row(
                str(i),
                f"{scored.score:.4f}",
                scored.chunk.source or scored.chunk.doc_id,
                scored.chunk.text[:300],
            )
        console.print(table)

    asyncio.run(run())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    top_k: int = typer.Option(6, help="Chunks to ground on."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output."),
) -> None:
    """Ask a question; answer is grounded in the index with citations."""

    async def run() -> None:
        engine = _engine()
        if no_stream:
            answer = await engine.ask(question, top_k=top_k)
            console.print(answer.text)
            citations = answer.citations
        else:
            citations = []
            async for event in engine.ask_stream(question, top_k=top_k):
                if event.type == "delta":
                    console.print(event.text, end="")
                elif event.answer:
                    citations = event.answer.citations
            console.print()
        if citations:
            console.print("\n[dim]sources:[/dim]")
            for citation in citations:
                console.print(f"  [dim][{citation.marker}] {citation.source or citation.doc_id}[/dim]")

    asyncio.run(run())


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host (default RAGNITE_HOST or 127.0.0.1)."),
    port: int = typer.Option(None, help="Bind port (default RAGNITE_PORT or 8000)."),
) -> None:
    """Run the HTTP API (requires `pip install ragnite[server]`)."""
    import uvicorn

    from ragnite.server.app import create_app

    cfg = RagniteConfig.from_env()
    uvicorn.run(create_app(config=cfg), host=host or cfg.host, port=port or cfg.port)


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio (requires `pip install ragnite[mcp]`)."""
    from ragnite.server.mcp import main as mcp_main

    mcp_main()


@app.command(name="eval")
def eval_cmd(
    dataset: str = typer.Argument(..., help="JSONL dataset: {query, relevant_ids, reference?}."),
    k: int = typer.Option(6, help="Cutoff for retrieval metrics."),
    judge: bool = typer.Option(False, help="Also grade answers with the configured LLM."),
) -> None:
    """Evaluate retrieval (hit@k, MRR, nDCG) and optionally generation quality."""

    async def run() -> None:
        from ragnite.eval.runner import load_dataset, run_eval

        report = await run_eval(_engine(), load_dataset(dataset), k=k, judge=judge)
        table = Table(title=f"Ragnite eval — {report.cases} cases @ k={report.k}")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("hit_rate", f"{report.hit_rate:.3f}")
        table.add_row("mrr", f"{report.mrr:.3f}")
        table.add_row("ndcg", f"{report.ndcg:.3f}")
        if report.faithfulness is not None:
            table.add_row("faithfulness", f"{report.faithfulness:.3f}")
        if report.answer_relevancy is not None:
            table.add_row("answer_relevancy", f"{report.answer_relevancy:.3f}")
        console.print(table)

    asyncio.run(run())


@app.command()
def stats() -> None:
    """Show index statistics."""

    async def run() -> None:
        console.print(json.dumps(await _engine().stats(), indent=2))

    asyncio.run(run())


@app.command()
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete every chunk in the current collection."""
    if not yes and not typer.confirm("Delete ALL chunks in this collection?"):
        raise typer.Abort()

    async def run() -> None:
        await _engine().clear()
        console.print("[green]collection cleared[/green]")

    asyncio.run(run())


if __name__ == "__main__":
    app()
