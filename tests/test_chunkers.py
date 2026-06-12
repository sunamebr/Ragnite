from ragnite.ingest.chunkers import CodeChunker, MarkdownChunker, RecursiveChunker, chunker_for
from ragnite.types import Document


def test_recursive_respects_size_budget():
    text = " ".join(f"word{i}" for i in range(400))
    doc = Document(text=text)
    chunks = RecursiveChunker(chunk_size=400, overlap=50).chunk(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 450 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == doc.id for c in chunks)


def test_recursive_keeps_short_text_whole():
    doc = Document(text="short text")
    chunks = RecursiveChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_markdown_heading_paths():
    doc = Document(
        text="# Guide\n\nintro paragraph\n\n## Install\n\npip install ragnite\n\n## Usage\n\nrun it",
        metadata={"suffix": ".md"},
    )
    chunks = MarkdownChunker().chunk(doc)
    headings = [c.metadata.get("heading") for c in chunks]
    assert any(h == "Guide > Install" for h in headings)
    assert any(h == "Guide > Usage" for h in headings)


def test_code_chunker_splits_on_definitions():
    body = "\n".join(f"    x{i} = {i}" for i in range(30))
    text = f"def alpha():\n{body}\n\ndef beta():\n{body}\n"
    chunks = CodeChunker(chunk_size=600, overlap=0).chunk(Document(text=text))
    assert len(chunks) >= 2


def test_chunker_dispatch():
    md = Document(text="x", metadata={"suffix": ".md"})
    py = Document(text="x", metadata={"suffix": ".py"})
    txt = Document(text="x", metadata={"suffix": ".txt"})
    assert isinstance(chunker_for(md), MarkdownChunker)
    assert isinstance(chunker_for(py), CodeChunker)
    assert isinstance(chunker_for(txt), RecursiveChunker)
