"""Secret redaction and sensitive-path filtering.

Everything Ragnite stores from a live Claude Code session (prompts used as
cache keys, episode texts, ingested docs) passes through ``redact()`` first.
Sensitive files are never ingested at all.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        "[REDACTED PRIVATE KEY]",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED]"),  # OpenAI / Anthropic style keys
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED]"),  # AWS access key id
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "[REDACTED]"),  # GitHub
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}\b"), "[REDACTED]"),  # GitLab
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED]"),  # Slack
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        "[REDACTED JWT]",
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer [REDACTED]"),
    (
        # credentials embedded in connection URLs: scheme://user:password@host
        re.compile(r"\b([A-Za-z][\w+.-]*://[^:/\s@]+):([^@\s]{3,})@"),
        r"\1:[REDACTED]@",
    ),
    (
        # key=value / key: value assignments for credential-ish names
        re.compile(
            r"(?i)\b(api[_-]?key|secret[_-]?key|secret|access[_-]?token|auth[_-]?token"
            r"|token|password|passwd|client[_-]?secret)\b(\s*[:=]\s*)(['\"]?)(\S{8,})"
        ),
        r"\1\2\3[REDACTED]",
    ),
]

_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".der", ".jks", ".keystore", ".kdbx"}
_SENSITIVE_BASENAMES = {
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
}
_SENSITIVE_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", ".env")


def redact(text: str) -> str:
    """Replace secret-looking material with placeholders."""
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern, _ in SECRET_PATTERNS)


def is_sensitive_path(path: str | Path) -> bool:
    """Files that must never be ingested, indexed, or quoted into memory."""
    name = Path(path).name.lower()
    if name in _SENSITIVE_BASENAMES:
        return True
    if any(name.startswith(prefix) for prefix in _SENSITIVE_PREFIXES):
        return True
    if Path(name).suffix in _SENSITIVE_SUFFIXES:
        return True
    return "cookie" in name or "ssh" in Path(path).parts


def load_ragniteignore(root: str | Path) -> list[str]:
    """Read .ragniteignore (gitignore-lite: one fnmatch glob per line, # comments)."""
    ignore_file = Path(root) / ".ragniteignore"
    if not ignore_file.exists():
        return []
    patterns: list[str] = []
    for line in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().rstrip("/")
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def is_ignored(rel_posix: str, patterns: list[str]) -> bool:
    """Match a root-relative posix path against .ragniteignore patterns.

    A pattern matches the full relative path, any parent directory, or the
    basename (so ``secret_notes.md``, ``private/*`` and ``private`` all work).
    """
    if not patterns:
        return False
    parts = rel_posix.split("/")
    for pattern in patterns:
        if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(parts[-1], pattern):
            return True
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if fnmatch.fnmatch(prefix, pattern):
                return True
    return False
