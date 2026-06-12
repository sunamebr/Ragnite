"""Memory subsystem data types.

Ragnite's memory layer is *typed*: a record is not just text, it carries kind,
subject, authority, lifecycle status and timestamps — the raw material the
confidence scorer needs to tell an LLM how much to trust what it recalls.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ragnite.types import new_id


class MemoryKind(StrEnum):
    FACT = "fact"  # stable truths about the project / product / domain / org
    DECISION = "decision"  # architectural & strategic decisions already made
    EPISODE = "episode"  # dev events: bugs fixed, failed attempts, progress
    CODE = "code"  # indexed repository: files, symbols, deps, endpoints


DEFAULT_AUTHORITY: dict[MemoryKind, float] = {
    MemoryKind.FACT: 0.80,
    MemoryKind.DECISION: 0.90,
    MemoryKind.EPISODE: 0.60,
    MemoryKind.CODE: 0.70,
}

RecordStatus = Literal["active", "superseded", "deprecated"]


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    kind: MemoryKind = MemoryKind.FACT
    text: str
    subject: str | None = None  # topic key, e.g. "db-port" or "src/auth.py::login"
    tags: list[str] = Field(default_factory=list)
    source: str | None = None  # where this came from (file, ADR, ticket, chat)
    authority: float = 0.7  # 0..1 trust in the source itself
    status: RecordStatus = "active"
    supersedes: str | None = None  # id of the record this one replaces
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.updated_at) / 86400.0)


class Evidence(BaseModel):
    """A recalled record plus retrieval signals."""

    record: MemoryRecord
    similarity: float  # normalized 0..1 (cosine, or squashed BM25)
    rank: int = 0
    in_dense: bool = False
    in_bm25: bool = False


AnswerMode = Literal["direct", "cautious", "ask_clarification", "search_more", "refuse_guess"]

MODE_SUGGESTIONS: dict[str, str] = {
    "direct": (
        "Strong consolidated memory. Answer directly from the provided context — "
        "do not re-derive, re-read sources, or re-analyze the project."
    ),
    "cautious": (
        "Partial evidence. Answer, but state caveats explicitly and attribute each "
        "claim to the memory entry it comes from."
    ),
    "ask_clarification": (
        "Memory holds conflicting or ambiguous entries on this topic. Ask the user "
        "one targeted clarifying question before answering."
    ),
    "search_more": (
        "Memory is insufficient. Retrieve more context (code search, docs, web) "
        "before answering — do not answer from these fragments alone."
    ),
    "refuse_guess": ("No reliable memory basis. Say you don't know rather than guessing."),
}


class ConfidenceSignals(BaseModel):
    top_similarity: float = 0.0
    mean_similarity: float = 0.0
    source_count: float = 0.0  # saturating signal, not a raw count
    agreement: float = 0.0  # dense & keyword retrieval agreeing on the same records
    recency: float = 0.0  # half-life decayed freshness
    authority: float = 0.0  # source trust
    conflict: bool = False


class ConfidenceReport(BaseModel):
    score: float
    signals: ConfidenceSignals
    rationale: list[str] = Field(default_factory=list)


class PackedContext(BaseModel):
    text: str = ""
    used: int = 0
    tokens: int = 0
    truncated: bool = False


class MemoryAnswer(BaseModel):
    """What an agent gets back from one ``recall()`` call: the smallest useful
    context plus an explicit conviction verdict."""

    query: str
    mode: AnswerMode
    confidence: float
    suggestion: str
    context: str = ""
    tokens: int = 0
    evidence: list[Evidence] = Field(default_factory=list)
    signals: ConfidenceSignals = Field(default_factory=ConfidenceSignals)
    cached: bool = False


class CodeIndexStats(BaseModel):
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    symbols: int = 0
