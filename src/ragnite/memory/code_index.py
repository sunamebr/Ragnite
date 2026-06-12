"""CodeMemory — indexed memory of a repository.

Extracts files, symbols (functions, classes, methods), imports/dependencies,
tests and HTTP endpoints into ``kind=code`` memory records. Python is parsed
with ``ast``; other languages use a definition-boundary regex. Indexing is
incremental: unchanged files (content hash) are skipped, changed files are
re-indexed, deleted files are evicted.

This is what stops a coding agent from re-reading the repo every session:
"where is auth handled?" becomes one recall against consolidated memory.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from ragnite.ingest.loaders import IGNORED_DIRS
from ragnite.memory.bank import MemoryBank
from ragnite.memory.types import CodeIndexStats, MemoryKind, MemoryRecord

CODE_SUFFIXES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".sql": "sql",
}

_GENERIC_SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:async\s+)?"
    r"(?:function|fn|func|class|interface|struct|enum|trait|impl|type|module)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)
_JS_CONST_FN = re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\(", re.MULTILINE)
_ROUTE = re.compile(r"@\w+\.(get|post|put|delete|patch|websocket)\(\s*['\"]([^'\"]+)")
_IMPORT_GENERIC = re.compile(
    r"^\s*(?:import\s+.*?from\s+['\"]([^'\"]+)|import\s+['\"]([^'\"]+)|"
    r"require\(\s*['\"]([^'\"]+)|use\s+([\w:]+))",
    re.MULTILINE,
)


def _file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _is_test_file(rel_path: str) -> bool:
    name = Path(rel_path).name.lower()
    return name.startswith("test_") or name.endswith("_test.py") or ".test." in name or ".spec." in name


class _Symbol:
    def __init__(self, name: str, symbol_type: str, line: int, snippet: str, route: str | None = None):
        self.name = name
        self.symbol_type = symbol_type
        self.line = line
        self.snippet = snippet
        self.route = route


def _python_symbols(source: str) -> tuple[list[_Symbol], list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    def route_of(node: ast.AST) -> str | None:
        for decorator in getattr(node, "decorator_list", []):
            segment = ast.get_source_segment(source, decorator) or ""
            match = _ROUTE.search(f"@{segment}")
            if match:
                return f"{match.group(1).upper()} {match.group(2)}"
        return None

    def snippet_of(node: ast.AST) -> str:
        doc = ast.get_docstring(node, clean=True) or ""
        segment = (ast.get_source_segment(source, node) or "").split("\n")
        signature = segment[0][:200] if segment else ""
        return f"{signature}\n{doc[:300]}".strip()

    symbols: list[_Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_Symbol(node.name, "function", node.lineno, snippet_of(node), route_of(node)))
        elif isinstance(node, ast.ClassDef):
            symbols.append(_Symbol(node.name, "class", node.lineno, snippet_of(node)))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        _Symbol(
                            f"{node.name}.{child.name}",
                            "method",
                            child.lineno,
                            snippet_of(child),
                            route_of(child),
                        )
                    )
    return symbols, sorted(set(imports))


def _generic_symbols(source: str) -> tuple[list[_Symbol], list[str]]:
    symbols: list[_Symbol] = []
    for pattern, symbol_type in ((_GENERIC_SYMBOL, "symbol"), (_JS_CONST_FN, "function")):
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            snippet = source[match.start() : match.start() + 200].split("\n")[0]
            symbols.append(_Symbol(match.group(1), symbol_type, line, snippet))
    imports = sorted({g for match in _IMPORT_GENERIC.finditer(source) for g in match.groups() if g})
    return symbols, imports


class CodeMemory:
    def __init__(self, bank: MemoryBank, max_file_bytes: int = 512_000) -> None:
        self.bank = bank
        self.max_file_bytes = max_file_bytes

    async def _existing(self) -> dict[str, tuple[str, list[str]]]:
        """Map of indexed file -> (content hash, record ids)."""
        index: dict[str, tuple[str, list[str]]] = {}
        for record in await self.bank.list(kind=MemoryKind.CODE):
            file = record.metadata.get("file")
            if not file:
                continue
            file_hash, ids = index.get(file, (record.metadata.get("file_hash", ""), []))
            ids.append(record.id)
            index[file] = (record.metadata.get("file_hash", file_hash), ids)
        return index

    def _records_for(self, rel_path: str, source: str, language: str) -> list[MemoryRecord]:
        file_hash = _file_hash(source)
        is_test = _is_test_file(rel_path)
        symbols, imports = _python_symbols(source) if language == "python" else _generic_symbols(source)

        common = {"file": rel_path, "file_hash": file_hash, "language": language}
        records: list[MemoryRecord] = []

        symbol_names = ", ".join(s.name for s in symbols[:60]) or "(no symbols detected)"
        import_names = ", ".join(imports[:40]) or "(none)"
        records.append(
            MemoryRecord(
                kind=MemoryKind.CODE,
                subject=rel_path,
                text=(
                    f"file {rel_path} ({language}, {len(symbols)} symbols"
                    f"{', test file' if is_test else ''})\n"
                    f"imports: {import_names}\nsymbols: {symbol_names}"
                ),
                source=rel_path,
                tags=["file"] + (["test"] if is_test else []),
                metadata={**common, "symbol_type": "file", "imports": imports},
            )
        )
        for symbol in symbols:
            tags = [symbol.symbol_type] + (["test"] if is_test else [])
            metadata = {
                **common,
                "symbol": symbol.name,
                "symbol_type": symbol.symbol_type,
                "line": symbol.line,
            }
            if symbol.route:
                tags.append("endpoint")
                metadata["route"] = symbol.route
            route_note = f" [{symbol.route}]" if symbol.route else ""
            records.append(
                MemoryRecord(
                    kind=MemoryKind.CODE,
                    subject=f"{rel_path}::{symbol.name}",
                    text=f"{symbol.symbol_type} {symbol.name}{route_note} — {rel_path}:{symbol.line}\n{symbol.snippet}",
                    source=f"{rel_path}:{symbol.line}",
                    tags=tags,
                    metadata=metadata,
                )
            )
        return records

    async def index_repo(self, path: str | Path) -> CodeIndexStats:
        root = Path(path)
        existing = await self._existing()
        stats = CodeIndexStats()
        seen_files: set[str] = set()

        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in CODE_SUFFIXES:
                continue
            if any(part in IGNORED_DIRS for part in candidate.parts):
                continue
            if candidate.stat().st_size > self.max_file_bytes:
                continue
            rel_path = candidate.relative_to(root).as_posix() if root.is_dir() else candidate.name
            seen_files.add(rel_path)
            source = candidate.read_text(encoding="utf-8", errors="replace")

            previous = existing.get(rel_path)
            if previous and previous[0] == _file_hash(source):
                stats.files_skipped += 1
                continue
            if previous:
                await self.bank.delete(previous[1])

            records = self._records_for(rel_path, source, CODE_SUFFIXES[candidate.suffix.lower()])
            await self.bank.add(records)
            stats.files_indexed += 1
            stats.symbols += max(0, len(records) - 1)

        for stale_file, (_, ids) in existing.items():
            if stale_file not in seen_files:
                await self.bank.delete(ids)
                stats.files_removed += 1
        return stats

    async def graph(self) -> dict[str, list[str]]:
        """File -> imports map from the indexed records (module relations)."""
        edges: dict[str, list[str]] = {}
        for record in await self.bank.list(kind=MemoryKind.CODE):
            if record.metadata.get("symbol_type") == "file":
                edges[record.metadata["file"]] = list(record.metadata.get("imports", []))
        return edges
