"""Chunking strategies.

Sizes are expressed in characters (~4 chars per token for latin scripts).
All chunkers guarantee non-empty chunks and stable ``Chunk.index`` ordering.
"""

from __future__ import annotations

import re
from typing import Protocol

from ragnite.types import Chunk, Document

DEFAULT_CHUNK_SIZE = 1600
DEFAULT_OVERLAP = 200


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


def _make_chunks(doc: Document, pieces: list[str], extra_meta: list[dict] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        text = piece.strip()
        if not text:
            continue
        meta = dict(doc.metadata)
        if extra_meta and i < len(extra_meta):
            meta.update(extra_meta[i])
        chunks.append(Chunk(doc_id=doc.id, text=text, index=len(chunks), source=doc.source, metadata=meta))
    return chunks


class RecursiveChunker:
    """Split on paragraph > line > sentence > word boundaries, packing to a size budget."""

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        return _make_chunks(doc, self.split(doc.text))

    def split(self, text: str) -> list[str]:
        fragments = self._split(text, 0)
        return self._pack(fragments)

    def _split(self, text: str, level: int) -> list[str]:
        if len(text) <= self.chunk_size or level >= len(self.SEPARATORS):
            return [text] if text.strip() else []
        sep = self.SEPARATORS[level]
        parts = [p + (sep if i < len(text.split(sep)) - 1 else "") for i, p in enumerate(text.split(sep))]
        out: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                out.extend(self._split(part, level + 1))
            elif part.strip():
                out.append(part)
        return out

    def _pack(self, fragments: list[str]) -> list[str]:
        packed: list[str] = []
        buf = ""
        for frag in fragments:
            if buf and len(buf) + len(frag) > self.chunk_size:
                packed.append(buf)
                buf = buf[-self.overlap :] if self.overlap else ""
            buf += frag
        if buf.strip():
            packed.append(buf)
        return packed


class MarkdownChunker:
    """Heading-aware splitter: sections become chunks, oversized sections recurse.

    Each chunk records its heading path in ``metadata["heading"]``.
    """

    _HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
        self._inner = RecursiveChunker(chunk_size, overlap)
        self.chunk_size = chunk_size

    def chunk(self, doc: Document) -> list[Chunk]:
        sections = self._sections(doc.text)
        pieces: list[str] = []
        metas: list[dict] = []
        for heading, body in sections:
            if not body.strip():
                continue
            if len(body) > self.chunk_size:
                for sub in self._inner.split(body):
                    pieces.append(sub)
                    metas.append({"heading": heading} if heading else {})
            else:
                pieces.append(body)
                metas.append({"heading": heading} if heading else {})
        return _make_chunks(doc, pieces, metas)

    def _sections(self, text: str) -> list[tuple[str, str]]:
        matches = list(self._HEADING.finditer(text))
        if not matches:
            return [("", text)]
        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()]
        if preamble.strip():
            sections.append(("", preamble))
        path: dict[int, str] = {}
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            path[level] = title
            for deeper in [k for k in path if k > level]:
                del path[deeper]
            heading = " > ".join(path[k] for k in sorted(path))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((heading, text[match.start() : end]))
        return sections


class CodeChunker:
    """Split source code on top-level definitions, falling back to recursive packing."""

    _BOUNDARY = re.compile(
        r"^(?=(?:export\s+)?(?:async\s+)?(?:def|class|function|fn|func|impl|interface|struct|enum|type|pub fn)\b)",
        re.MULTILINE,
    )

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
        self._inner = RecursiveChunker(chunk_size, overlap)
        self.chunk_size = chunk_size

    def chunk(self, doc: Document) -> list[Chunk]:
        blocks = self._BOUNDARY.split(doc.text)
        pieces: list[str] = []
        for block in blocks:
            if len(block) > self.chunk_size:
                pieces.extend(self._inner.split(block))
            elif block.strip():
                pieces.append(block)
        return _make_chunks(doc, pieces)


_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".lua",
}


def chunker_for(
    doc: Document, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> Chunker:
    """Pick a chunker based on the document's source suffix."""
    suffix = (doc.metadata.get("suffix") or "").lower()
    if not suffix and doc.source and "." in doc.source:
        suffix = "." + doc.source.rsplit(".", 1)[-1].lower()
    if suffix in {".md", ".markdown"}:
        return MarkdownChunker(chunk_size, overlap)
    if suffix in _CODE_SUFFIXES:
        return CodeChunker(chunk_size, overlap)
    return RecursiveChunker(chunk_size, overlap)
