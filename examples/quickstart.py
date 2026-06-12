"""Ragnite quickstart.

Zero-config run (keyword search only):
    python examples/quickstart.py

Full run (semantic search + grounded answers):
    set VOYAGE_API_KEY=...      # embeddings
    set ANTHROPIC_API_KEY=...   # generation
    python examples/quickstart.py
"""

import asyncio
from pathlib import Path

from ragnite import build_engine

DATA = Path(__file__).parent / "data"


async def main() -> None:
    engine = build_engine()
    stats = await engine.ingest_path(DATA)
    print(f"indexed {stats.chunks} chunks from {stats.documents} documents\n")

    query = "Why does Mars appear red?"
    print(f"search: {query}")
    for scored in await engine.search(query, top_k=3):
        print(f"  {scored.score:.4f}  {scored.chunk.source}: {scored.chunk.text[:80]}...")

    if engine.llm is None:
        print("\n(no LLM configured — set ANTHROPIC_API_KEY to enable grounded answers)")
        return

    print("\nstreaming answer:")
    async for event in engine.ask_stream(query):
        if event.type == "delta":
            print(event.text, end="", flush=True)
        elif event.answer:
            print("\n\nsources:")
            for citation in event.answer.citations:
                print(f"  [{citation.marker}] {citation.source}")


if __name__ == "__main__":
    asyncio.run(main())
