import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import ragnite
from conftest import FakeChat, sample_docs
from ragnite.config import RagniteConfig
from ragnite.embed.fake import FakeEmbedder
from ragnite.rag.engine import RagEngine
from ragnite.server.app import create_app
from ragnite.store.native import NativeVectorStore


def _config(tmp_path, **overrides) -> RagniteConfig:
    return RagniteConfig(data_dir=tmp_path / "data", embedder="fake", llm="none", **overrides)


def _client(tmp_path, **overrides) -> TestClient:
    engine = RagEngine(
        store=NativeVectorStore(tmp_path / "collection"),
        embedder=FakeEmbedder(),
        llm=FakeChat(),
    )
    app = create_app(engine=engine, config=_config(tmp_path, **overrides))
    return TestClient(app)


def test_app_version_matches_package(tmp_path):
    client = _client(tmp_path)
    assert client.app.version == ragnite.__version__
    schema = client.get("/openapi.json").json()
    assert schema["info"]["version"] == ragnite.__version__


def test_healthz(tmp_path):
    response = _client(tmp_path).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_search_ask_roundtrip(tmp_path):
    client = _client(tmp_path)
    docs = [{"id": d.id, "text": d.text, "source": d.source, "metadata": d.metadata} for d in sample_docs()]
    ingest = client.post("/v1/ingest", json={"documents": docs})
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] >= 3

    search = client.post("/v1/search", json={"query": "why is mars red", "top_k": 3})
    assert search.status_code == 200
    results = search.json()["results"]
    assert results and results[0]["chunk"]["doc_id"] == "doc_mars"

    ask = client.post("/v1/ask", json={"query": "why is mars red?"})
    assert ask.status_code == 200
    body = ask.json()
    assert "iron oxide" in body["text"]
    assert body["citations"]


def test_memory_remember_and_recall_endpoints(tmp_path):
    client = _client(tmp_path)
    remember = client.post(
        "/v1/memory/remember",
        json={"text": "Services talk over gRPC.", "kind": "decision", "subject": "api-style"},
    )
    assert remember.status_code == 200
    record = remember.json()
    assert record["kind"] == "decision" and record["id"].startswith("mem_")

    recall = client.post("/v1/memory/recall", json={"query": "how do services communicate?"})
    assert recall.status_code == 200
    answer = recall.json()
    assert {"mode", "confidence", "context", "suggestion", "signals"} <= set(answer)
    assert answer["mode"] in {"direct", "cautious", "ask_clarification", "search_more", "refuse_guess"}
    assert "gRPC" in answer["context"]

    stats = client.get("/v1/memory/stats")
    assert stats.status_code == 200
    assert stats.json()["active_by_kind"]["decision"] == 1


def test_memory_index_code_endpoint(tmp_path):
    client = _client(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def entrypoint():\n    pass\n", encoding="utf-8")
    response = client.post("/v1/memory/index_code", json={"path": str(repo)})
    assert response.status_code == 200
    assert response.json()["files_indexed"] == 1


def test_bearer_auth_enforced(tmp_path):
    client = _client(tmp_path, api_key="sekret")
    denied = client.post("/v1/search", json={"query": "x"})
    assert denied.status_code == 401
    allowed = client.post("/v1/search", json={"query": "x"}, headers={"Authorization": "Bearer sekret"})
    assert allowed.status_code == 200
    assert client.get("/healthz").status_code == 200  # liveness stays open
