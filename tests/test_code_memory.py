from ragnite.embed.fake import FakeEmbedder
from ragnite.memory.bank import MemoryBank
from ragnite.memory.code_index import CodeMemory
from ragnite.memory.types import MemoryKind

APP_PY = '''import os
from fastapi import FastAPI

app = FastAPI()


@app.get("/users")
def list_users():
    """Return all users."""
    return []


class UserService:
    def create(self, name):
        return name


def helper():
    pass
'''

UTIL_JS = """export function renderWidget() {}
const computeTotal = (a, b) => a + b
"""

TEST_PY = """def test_smoke():
    assert True
"""


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text(APP_PY, encoding="utf-8")
    (repo / "util.js").write_text(UTIL_JS, encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(TEST_PY, encoding="utf-8")
    return repo


def _code_memory(tmp_path) -> CodeMemory:
    return CodeMemory(MemoryBank(embedder=FakeEmbedder(), path=tmp_path / "bank"))


async def test_index_extracts_symbols_endpoints_and_tests(tmp_path):
    code = _code_memory(tmp_path)
    stats = await code.index_repo(_repo(tmp_path))
    assert stats.files_indexed == 3
    assert (
        stats.symbols >= 6
    )  # list_users, UserService, .create, helper, renderWidget, computeTotal, test_smoke

    records = await code.bank.list(kind=MemoryKind.CODE)
    by_symbol = {r.metadata.get("symbol"): r for r in records if r.metadata.get("symbol")}

    endpoint = by_symbol["list_users"]
    assert endpoint.metadata["route"] == "GET /users"
    assert "endpoint" in endpoint.tags

    assert "UserService.create" in by_symbol
    assert by_symbol["renderWidget"].metadata["language"] == "javascript"
    assert "test" in by_symbol["test_smoke"].tags

    app_file = next(
        r for r in records if r.metadata.get("symbol_type") == "file" and r.metadata["file"] == "app.py"
    )
    assert "fastapi" in app_file.metadata["imports"]


async def test_incremental_reindex_and_eviction(tmp_path):
    code = _code_memory(tmp_path)
    repo = _repo(tmp_path)
    await code.index_repo(repo)

    # nothing changed -> everything skipped
    second = await code.index_repo(repo)
    assert second.files_indexed == 0
    assert second.files_skipped == 3

    # change one file -> only it is re-indexed, without duplicate records
    (repo / "app.py").write_text(APP_PY + "\n\ndef another():\n    pass\n", encoding="utf-8")
    third = await code.index_repo(repo)
    assert third.files_indexed == 1
    assert third.files_skipped == 2
    records = await code.bank.list(kind=MemoryKind.CODE)
    app_file_records = [
        r for r in records if r.metadata.get("symbol_type") == "file" and r.metadata["file"] == "app.py"
    ]
    assert len(app_file_records) == 1
    assert any(r.metadata.get("symbol") == "another" for r in records)

    # delete a file -> its records are evicted
    (repo / "util.js").unlink()
    fourth = await code.index_repo(repo)
    assert fourth.files_removed == 1
    records = await code.bank.list(kind=MemoryKind.CODE)
    assert all(r.metadata.get("file") != "util.js" for r in records)


async def test_import_graph(tmp_path):
    code = _code_memory(tmp_path)
    await code.index_repo(_repo(tmp_path))
    graph = await code.graph()
    assert "fastapi" in graph["app.py"]
    assert "os" in graph["app.py"]


async def test_recall_finds_endpoint_by_question(tmp_path):
    code = _code_memory(tmp_path)
    await code.index_repo(_repo(tmp_path))
    evidence = await code.bank.recall("where are the users listed endpoint", k=5)
    assert evidence
    top_subjects = [e.record.subject or "" for e in evidence[:3]]
    assert any("list_users" in s or s == "app.py" for s in top_subjects)
