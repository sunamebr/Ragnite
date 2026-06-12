"""FastAPI HTTP service (optional extra ``ragnite[server]``).

Endpoints:
    GET  /healthz          liveness
    GET  /v1/stats         index statistics
    POST /v1/ingest        ingest raw documents
    POST /v1/search        hybrid retrieval
    POST /v1/ask           grounded answer; ``stream=true`` returns SSE

Auth: set RAGNITE_API_KEY to require ``Authorization: Bearer <key>``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ragnite.config import RagniteConfig, build_engine
from ragnite.errors import MissingDependencyError
from ragnite.rag.engine import RagEngine

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None


class IngestRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(description="Items with 'text' and optional 'source'/'metadata'.")


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None


class AskRequest(SearchRequest):
    stream: bool = False


def create_app(engine: RagEngine | None = None, config: RagniteConfig | None = None):
    if FastAPI is None:
        raise MissingDependencyError("fastapi", "server")

    cfg = config or RagniteConfig.from_env()
    rag = engine or build_engine(cfg)
    app = FastAPI(title="Ragnite", version="0.1.0", description="Production-grade RAG API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    async def require_auth(request: Request) -> None:
        if cfg.api_key is None:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {cfg.api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/stats", dependencies=[Depends(require_auth)])
    async def stats() -> dict:
        return await rag.stats()

    @app.post("/v1/ingest", dependencies=[Depends(require_auth)])
    async def ingest(body: IngestRequest) -> dict:
        from ragnite.types import Document

        docs = [
            Document(
                text=item["text"],
                source=item.get("source"),
                metadata=item.get("metadata") or {},
            )
            for item in body.documents
            if item.get("text", "").strip()
        ]
        if not docs:
            raise HTTPException(status_code=422, detail="no non-empty documents provided")
        result = await rag.ingest_documents(docs)
        return result.model_dump()

    @app.post("/v1/search", dependencies=[Depends(require_auth)])
    async def search(body: SearchRequest) -> dict:
        results = await rag.search(body.query, top_k=body.top_k, filters=body.filters)
        return {"results": [r.model_dump() for r in results]}

    @app.post("/v1/ask", dependencies=[Depends(require_auth)])
    async def ask(body: AskRequest):
        if not body.stream:
            answer = await rag.ask(body.query, top_k=body.top_k, filters=body.filters)
            return answer.model_dump()

        async def event_stream():
            async for event in rag.ask_stream(body.query, top_k=body.top_k, filters=body.filters):
                yield f"data: {json.dumps(event.model_dump(), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
