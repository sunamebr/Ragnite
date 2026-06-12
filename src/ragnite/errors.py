"""Ragnite exception hierarchy."""

from __future__ import annotations


class RagniteError(Exception):
    """Base class for all Ragnite errors."""


class ConfigError(RagniteError):
    """Invalid or missing configuration (e.g. no embedder where one is required)."""


class MissingDependencyError(RagniteError):
    """An optional dependency is required for this feature."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"'{package}' is required for this feature. Install it with: pip install ragnite[{extra}]"
        )


class IngestionError(RagniteError):
    """A document could not be loaded or chunked."""


class RetrievalError(RagniteError):
    """Search failed."""


class GenerationError(RagniteError):
    """The LLM call failed or was refused."""
