import pytest

from ragnite.errors import IngestionError
from ragnite.ingest.loaders import load_path, load_text


def test_load_directory_skips_ignored(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nhello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "ignored.md").write_text("nope", encoding="utf-8")

    docs = load_path(tmp_path)
    names = sorted(d.metadata["filename"] for d in docs)
    assert names == ["a.md", "b.txt", "c.py"]
    assert all(d.source for d in docs)


def test_load_html_strips_tags(tmp_path):
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><style>x{}</style></head><body><h1>Hello</h1><p>World</p>"
        "<script>evil()</script></body></html>",
        encoding="utf-8",
    )
    docs = load_path(page)
    assert len(docs) == 1
    assert "Hello" in docs[0].text and "World" in docs[0].text
    assert "evil" not in docs[0].text


def test_load_missing_path_raises():
    with pytest.raises(IngestionError):
        load_path("does/not/exist")


def test_load_text_wraps_document():
    doc = load_text("hello", source="note", metadata={"k": "v"})
    assert doc.text == "hello"
    assert doc.source == "note"
    assert doc.metadata == {"k": "v"}
