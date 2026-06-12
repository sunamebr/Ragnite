"""Ragnite CLI — confidence-aware RAG memory engine for LLMs and coding agents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragnite.config import RagniteConfig, build_engine

app = typer.Typer(
    name="ragnite",
    help=(
        "Confidence-aware RAG memory engine for LLMs and coding agents. "
        "Memory: remember / recall (verdict + packed context) / index-code. "
        "Document RAG: ingest / query / ask. Serve over HTTP or MCP; evaluate with eval."
    ),
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


@app.command()
def remember(
    text: str = typer.Argument(..., help="The knowledge to store."),
    kind: str = typer.Option("fact", help="fact | decision | episode"),
    subject: str = typer.Option(None, help="Topic key (e.g. 'db-port') — enables conflict detection."),
    source: str = typer.Option(None, help="Provenance (file, ADR, ticket)."),
    supersedes: str = typer.Option(None, help="Memory id this entry replaces."),
) -> None:
    """Store a memory record (Factual / Decision / Episodic memory)."""

    async def run() -> None:
        from ragnite.config import build_memory_engine

        record = await build_memory_engine(RagniteConfig.from_env()).remember(
            text, kind=kind, subject=subject, source=source, supersedes=supersedes
        )
        label = f" (subject: {subject})" if subject else ""
        console.print(f"[green]+[/green] {record.kind.value} {record.id}{label}")

    asyncio.run(run())


_MODE_COLORS = {
    "direct": "green",
    "cautious": "yellow",
    "ask_clarification": "magenta",
    "search_more": "cyan",
    "refuse_guess": "red",
}


@app.command()
def recall(
    query: str = typer.Argument(..., help="What to recall."),
    budget: int = typer.Option(2000, help="Token budget for the packed context."),
    kinds: str = typer.Option(None, help="Comma-separated filter: fact,decision,episode,code."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the semantic cache."),
) -> None:
    """Recall from memory: packed context + confidence + answer mode."""

    async def run() -> None:
        from ragnite.config import build_memory_engine
        from ragnite.memory import MemoryKind

        kind_list = [MemoryKind(k.strip()) for k in kinds.split(",")] if kinds else None
        answer = await build_memory_engine(RagniteConfig.from_env()).recall(
            query, kinds=kind_list, budget_tokens=budget, use_cache=not no_cache
        )
        color = _MODE_COLORS[answer.mode]
        cached = "  [dim](cached)[/dim]" if answer.cached else ""
        console.print(
            f"mode: [{color}]{answer.mode}[/{color}]  confidence: {answer.confidence:.2f}  "
            f"tokens: {answer.tokens}{cached}"
        )
        console.print(f"[dim]{answer.suggestion}[/dim]")
        if answer.context:
            console.print(answer.context)

    asyncio.run(run())


@app.command(name="index-code")
def index_code(
    path: str = typer.Argument(".", help="Repository root to index into Code Memory."),
) -> None:
    """Index a repository into Code Memory (incremental — unchanged files skipped)."""

    async def run() -> None:
        from ragnite.config import build_memory_engine

        stats = await build_memory_engine(RagniteConfig.from_env()).index_repo(path)
        console.print(
            f"[green]+[/green] indexed {stats.files_indexed} file(s) / {stats.symbols} symbol(s); "
            f"skipped {stats.files_skipped} unchanged; removed {stats.files_removed} stale"
        )

    asyncio.run(run())


claude_app = typer.Typer(
    help=(
        "Claude Code integration: install the /ragnite skill + MCP + hooks, bootstrap the "
        "project (init), and control event-driven live context injection (invoke/pause)."
    ),
    no_args_is_help=True,
)
app.add_typer(claude_app, name="claude")


@claude_app.command(name="install")
def claude_install(
    path: str = typer.Option(".", help="Project root to install into."),
) -> None:
    """Install /ragnite skill, MCP server, hooks, and invoke-mode config."""
    from ragnite.claude.installer import install_into

    for action in install_into(path):
        console.print(f"[green]+[/green] {action}")
    console.print(
        "\nNext: restart Claude Code in this project, then run [bold]/ragnite init[/bold] "
        "and [bold]/ragnite invoke[/bold]."
    )


@claude_app.command(name="init")
def claude_init(
    path: str = typer.Option(None, help="Project root (default: auto-detect)."),
) -> None:
    """Bootstrap: index code + docs, seed inferred memories, smoke recall."""
    from ragnite.claude.bootstrap import run_init
    from ragnite.claude.session import find_project_root

    root = Path(path) if path else find_project_root()
    console.print(f"bootstrapping [bold]{root}[/bold] ...")
    stats = asyncio.run(run_init(root))
    code = stats["code"]
    console.print(
        f"[green]+[/green] code: {code['files_indexed']} files / {code['symbols']} symbols "
        f"(skipped {code['files_skipped']} unchanged)"
    )
    console.print(f"[green]+[/green] docs: {stats['doc_chunks']} chunks from {stats['doc_files']} files")
    console.print(
        f"[green]+[/green] seeded {stats['seeded_inferred']} inferred memories "
        f"(tagged 'inferred' — confirm or correct them)"
    )
    for smoke in stats["smoke"]:
        console.print(
            f"  smoke recall: [dim]{smoke['query']}[/dim] -> {smoke['mode']} ({smoke['confidence']:.2f})"
        )
    counts = stats["memory"]["active_by_kind"]
    console.print(
        f"memory: {counts['fact']} facts / {counts['decision']} decisions / "
        f"{counts['episode']} episodes / {counts['code']} code records"
    )
    console.print("\nRun [bold]/ragnite invoke[/bold] to activate live context injection.")


def _validate_install(root: Path) -> list[str]:
    import json as _json

    problems: list[str] = []
    mcp_file = root / ".mcp.json"
    try:
        servers = _json.loads(mcp_file.read_text(encoding="utf-8")).get("mcpServers", {})
        if "ragnite" not in servers:
            problems.append(".mcp.json has no 'ragnite' server")
    except (OSError, _json.JSONDecodeError):
        problems.append(".mcp.json missing or unreadable")
    settings_file = root / ".claude" / "settings.local.json"
    try:
        raw = settings_file.read_text(encoding="utf-8")
        if "ragnite.cli claude hook" not in raw:
            problems.append("hooks not installed in .claude/settings.local.json")
    except OSError:
        problems.append(".claude/settings.local.json missing")
    return problems


@claude_app.command(name="invoke")
def claude_invoke(
    path: str = typer.Option(None, help="Project root (default: auto-detect)."),
) -> None:
    """Activate invoke mode (event-driven live context injection) and print the briefing."""
    from ragnite.claude.bootstrap import project_config
    from ragnite.claude.hooks import _briefing
    from ragnite.claude.session import SessionState, find_project_root, load_invoke_config
    from ragnite.config import build_memory_engine

    root = Path(path) if path else find_project_root()
    for problem in _validate_install(root):
        console.print(f"[yellow]![/yellow] {problem} — run `ragnite claude install` and restart the session")

    state = SessionState(root)
    state.set_active(True)
    console.print(
        "[green]invoke mode: ACTIVE[/green] — context will be injected at session start, "
        "on each prompt, and after relevant tool calls."
    )

    engine = build_memory_engine(project_config(root))
    briefing = asyncio.run(_briefing(engine, load_invoke_config(root)))
    if briefing:
        console.print("\n" + briefing)
    else:
        console.print("[dim]no memory yet — run /ragnite init first[/dim]")
    console.print(
        "\n[dim]Reminder to the assistant: from now on, call the ragnite `recall` MCP tool "
        "before re-reading or re-analyzing this repository, and obey injected "
        "<ragnite-context> modes.[/dim]"
    )


@claude_app.command(name="pause")
def claude_pause(
    path: str = typer.Option(None, help="Project root (default: auto-detect)."),
) -> None:
    """Deactivate invoke mode (memory is kept; injection stops)."""
    from ragnite.claude.session import SessionState, find_project_root

    root = Path(path) if path else find_project_root()
    SessionState(root).set_active(False)
    console.print("[yellow]invoke mode: PAUSED[/yellow] — memory kept; run /ragnite invoke to resume.")


@claude_app.command(name="status")
def claude_status(
    path: str = typer.Option(None, help="Project root (default: auto-detect)."),
) -> None:
    """Show invoke-mode state, stats, and memory counts."""
    from ragnite.claude.bootstrap import project_config
    from ragnite.claude.session import SessionState, find_project_root
    from ragnite.config import build_memory_engine

    root = Path(path) if path else find_project_root()
    state = SessionState(root)

    async def run() -> dict:
        engine = build_memory_engine(project_config(root))
        return await engine.stats()

    info = {
        "root": str(root),
        "active": state.active,
        "stats": state.data.get("stats", {}),
        "install_problems": _validate_install(root),
        "memory": asyncio.run(run()),
    }
    console.print(json.dumps(info, indent=2, ensure_ascii=False))


@claude_app.command(name="hook", hidden=True)
def claude_hook(event: str = typer.Argument(...)) -> None:
    """Hook entrypoint (stdin JSON -> stdout JSON). Never fails the session."""
    import sys

    from ragnite.claude.hooks import run_hook

    output = run_hook(event, sys.stdin.read())
    if output:
        sys.stdout.write(output)


if __name__ == "__main__":
    app()
