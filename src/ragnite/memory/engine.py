"""MemoryEngine — the conviction layer.

One ``recall()`` call gives an agent everything it needs to decide how to
proceed: the smallest sufficient context, a confidence score with its signals,
and an explicit answer mode (direct / cautious / ask_clarification /
search_more / refuse_guess). The goal is to stop agents from burning tokens
re-analyzing what is already consolidated.
"""

from __future__ import annotations

from pathlib import Path

from ragnite.memory.bank import MemoryBank
from ragnite.memory.code_index import CodeMemory
from ragnite.memory.packer import ContextPacker
from ragnite.memory.scorer import ConfidenceScorer, decide_mode
from ragnite.memory.semcache import SemanticCache
from ragnite.memory.types import (
    DEFAULT_AUTHORITY,
    MODE_SUGGESTIONS,
    CodeIndexStats,
    MemoryAnswer,
    MemoryKind,
    MemoryRecord,
)

_CACHEABLE_MODES = {"direct", "cautious"}


class MemoryEngine:
    def __init__(
        self,
        bank: MemoryBank,
        scorer: ConfidenceScorer | None = None,
        packer: ContextPacker | None = None,
        cache: SemanticCache | None = None,
        default_budget_tokens: int = 2000,
    ) -> None:
        self.bank = bank
        self.scorer = scorer or ConfidenceScorer()
        self.packer = packer or ContextPacker(budget_tokens=default_budget_tokens)
        self.cache = cache
        self.code = CodeMemory(bank)
        self.default_budget_tokens = default_budget_tokens

    # -- write path -------------------------------------------------------------

    async def remember(
        self,
        text: str,
        kind: MemoryKind | str = MemoryKind.FACT,
        subject: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        authority: float | None = None,
        supersedes: str | None = None,
    ) -> MemoryRecord:
        kind = MemoryKind(kind)
        record = MemoryRecord(
            kind=kind,
            text=text.strip(),
            subject=subject,
            tags=tags or [],
            source=source,
            authority=authority if authority is not None else DEFAULT_AUTHORITY[kind],
        )
        if supersedes:
            await self.bank.supersede(supersedes, record)
        else:
            await self.bank.add([record])
        if self.cache is not None:
            await self.cache.clear()  # memory changed -> cached verdicts are stale
        return record

    async def remember_fact(self, text: str, subject: str | None = None, **kwargs) -> MemoryRecord:
        return await self.remember(text, MemoryKind.FACT, subject=subject, **kwargs)

    async def remember_decision(
        self, text: str, subject: str | None = None, supersedes: str | None = None, **kwargs
    ) -> MemoryRecord:
        return await self.remember(
            text, MemoryKind.DECISION, subject=subject, supersedes=supersedes, **kwargs
        )

    async def remember_episode(self, text: str, subject: str | None = None, **kwargs) -> MemoryRecord:
        return await self.remember(text, MemoryKind.EPISODE, subject=subject, **kwargs)

    async def forget(self, record_id: str) -> bool:
        removed = await self.bank.delete([record_id]) > 0
        if removed and self.cache is not None:
            await self.cache.clear()
        return removed

    async def index_repo(self, path: str | Path) -> CodeIndexStats:
        stats = await self.code.index_repo(path)
        if (stats.files_indexed or stats.files_removed) and self.cache is not None:
            await self.cache.clear()
        return stats

    # -- read path ----------------------------------------------------------------

    async def recall(
        self,
        query: str,
        kinds: list[MemoryKind] | None = None,
        budget_tokens: int | None = None,
        top_k: int = 12,
        use_cache: bool = True,
    ) -> MemoryAnswer:
        if use_cache and self.cache is not None:
            cached = await self.cache.get(query)
            if cached is not None:
                return cached

        evidence = await self.bank.recall(query, kinds=kinds, k=top_k)
        report = self.scorer.score(evidence)
        mode = decide_mode(report, self.scorer.policy)
        packed = self.packer.pack(evidence, budget_tokens or self.default_budget_tokens)

        answer = MemoryAnswer(
            query=query,
            mode=mode,
            confidence=report.score,
            suggestion=MODE_SUGGESTIONS[mode],
            context=packed.text,
            tokens=packed.tokens,
            evidence=evidence,
            signals=report.signals,
        )
        if use_cache and self.cache is not None and mode in _CACHEABLE_MODES:
            await self.cache.put(query, answer)
        return answer

    # -- introspection --------------------------------------------------------------

    async def stats(self) -> dict:
        by_kind = {kind.value: len(await self.bank.list(kind=kind)) for kind in MemoryKind}
        return {
            "records": await self.bank.count(),
            "active_by_kind": by_kind,
            "cache_entries": await self.cache.count() if self.cache else 0,
            "embedder": self.bank.embedder.name if self.bank.embedder else None,
            "policy": self.scorer.policy.model_dump(exclude={"half_life_days"}),
        }
