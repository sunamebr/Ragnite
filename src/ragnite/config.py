"""Environment-driven configuration and component factories.

Everything is overridable with ``RAGNITE_*`` variables; provider API keys use
their standard names (``ANTHROPIC_API_KEY``, ``VOYAGE_API_KEY``, ...). With no
configuration at all, Ragnite runs keyword-only (BM25) with the native store.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pydantic import BaseModel

if TYPE_CHECKING:
    from ragnite.memory.engine import MemoryEngine

from ragnite.embed.base import EmbeddingProvider
from ragnite.embed.cache import EmbeddingCache
from ragnite.errors import ConfigError
from ragnite.llm.base import ChatModel
from ragnite.rag.engine import RagEngine
from ragnite.rag.memory import VectorMemory
from ragnite.retrieve.rerank import Reranker
from ragnite.retrieve.retriever import RetrievalConfig
from ragnite.store.base import VectorStore


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class RagniteConfig(BaseModel):
    data_dir: Path = Path(".ragnite")
    collection: str = "default"

    embedder: str = "auto"  # auto | voyage | openai | local | fake | none
    embedding_model: str | None = None
    embedding_cache: bool = True

    llm: str = "auto"  # auto | anthropic | openai | none
    llm_model: str | None = None

    store: str = "native"  # native | qdrant
    qdrant_url: str = "http://localhost:6333"

    reranker: str = "none"  # none | cohere | voyage | llm
    contextual: bool = False
    chunk_size: int = 1600
    chunk_overlap: int = 200

    retrieval: RetrievalConfig = RetrievalConfig()

    # memory engine
    cache_threshold: float = 0.90  # semantic (verdict) cache similarity cutoff
    cache_ttl_days: float = 7.0
    memory_budget_tokens: int = 2000  # default context-packer budget
    answer_cache: bool = False  # opt-in final-answer cache for RagEngine.ask

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None  # bearer token for the HTTP API; None disables auth

    @classmethod
    def from_env(cls) -> RagniteConfig:
        load_dotenv()
        cfg = cls(
            data_dir=Path(_env("RAGNITE_DATA_DIR", ".ragnite")),
            collection=_env("RAGNITE_COLLECTION", "default"),
            embedder=_env("RAGNITE_EMBEDDER", "auto").lower(),
            embedding_model=_env("RAGNITE_EMBEDDING_MODEL") or None,
            embedding_cache=_env("RAGNITE_EMBEDDING_CACHE", "1") not in {"0", "false", "no"},
            llm=_env("RAGNITE_LLM", "auto").lower(),
            llm_model=_env("RAGNITE_LLM_MODEL") or None,
            store=_env("RAGNITE_STORE", "native").lower(),
            qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
            reranker=_env("RAGNITE_RERANKER", "none").lower(),
            contextual=_env("RAGNITE_CONTEXTUAL", "0") in {"1", "true", "yes"},
            chunk_size=int(_env("RAGNITE_CHUNK_SIZE", "1600")),
            chunk_overlap=int(_env("RAGNITE_CHUNK_OVERLAP", "200")),
            cache_threshold=float(_env("RAGNITE_CACHE_THRESHOLD", "0.90")),
            cache_ttl_days=float(_env("RAGNITE_CACHE_TTL_DAYS", "7")),
            memory_budget_tokens=int(_env("RAGNITE_MEMORY_BUDGET", "2000")),
            answer_cache=_env("RAGNITE_ANSWER_CACHE", "0") in {"1", "true", "yes"},
            host=_env("RAGNITE_HOST", "127.0.0.1"),
            port=int(_env("RAGNITE_PORT", "8000")),
            api_key=_env("RAGNITE_API_KEY") or None,
        )
        if top_k := _env("RAGNITE_TOP_K"):
            cfg.retrieval.top_k = int(top_k)
        return cfg


# -- factories -------------------------------------------------------------------


def build_embedder(cfg: RagniteConfig) -> EmbeddingProvider | None:
    kind = cfg.embedder
    if kind == "auto":
        if _env("VOYAGE_API_KEY"):
            kind = "voyage"
        elif _env("OPENAI_API_KEY"):
            kind = "openai"
        else:
            return None

    provider: EmbeddingProvider
    if kind == "none":
        return None
    if kind == "voyage":
        from ragnite.embed.voyage import VoyageEmbedder

        provider = VoyageEmbedder(model=cfg.embedding_model or "voyage-3.5")
    elif kind == "openai":
        from ragnite.embed.openai_compat import OpenAICompatEmbedder

        provider = OpenAICompatEmbedder(model=cfg.embedding_model or "text-embedding-3-small")
    elif kind == "local":
        from ragnite.embed.local import LocalEmbedder

        provider = LocalEmbedder(model=cfg.embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
    elif kind == "fake":
        from ragnite.embed.fake import FakeEmbedder

        provider = FakeEmbedder()
    else:
        raise ConfigError(f"unknown embedder: {kind!r}")

    if cfg.embedding_cache and kind != "fake":
        provider = EmbeddingCache(provider, cfg.data_dir / "embeddings.db")
    return provider


def build_llm(cfg: RagniteConfig) -> ChatModel | None:
    kind = cfg.llm
    if kind == "auto":
        if _env("ANTHROPIC_API_KEY"):
            kind = "anthropic"
        elif _env("OPENAI_API_KEY"):
            kind = "openai"
        else:
            return None
    if kind == "none":
        return None
    if kind == "anthropic":
        from ragnite.llm.anthropic import AnthropicChat

        return AnthropicChat(model=cfg.llm_model or "claude-opus-4-8")
    if kind == "openai":
        from ragnite.llm.openai_compat import OpenAICompatChat

        return OpenAICompatChat(model=cfg.llm_model or "gpt-4o")
    raise ConfigError(f"unknown llm: {kind!r}")


def build_store(cfg: RagniteConfig) -> VectorStore:
    if cfg.store == "native":
        from ragnite.store.native import NativeVectorStore

        return NativeVectorStore(cfg.data_dir / "collections" / cfg.collection)
    if cfg.store == "qdrant":
        from ragnite.store.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            url=cfg.qdrant_url,
            collection=f"ragnite_{cfg.collection}",
            api_key=_env("QDRANT_API_KEY") or None,
        )
    raise ConfigError(f"unknown store: {cfg.store!r}")


def build_reranker(cfg: RagniteConfig, llm: ChatModel | None) -> Reranker | None:
    kind = cfg.reranker
    if kind in {"", "none"}:
        return None
    if kind == "cohere":
        from ragnite.retrieve.rerank import CohereReranker

        return CohereReranker()
    if kind == "voyage":
        from ragnite.retrieve.rerank import VoyageReranker

        return VoyageReranker()
    if kind == "llm":
        if llm is None:
            raise ConfigError("reranker=llm requires a configured LLM")
        from ragnite.retrieve.rerank import LLMReranker

        return LLMReranker(llm)
    raise ConfigError(f"unknown reranker: {kind!r}")


def build_engine(cfg: RagniteConfig | None = None) -> RagEngine:
    cfg = cfg or RagniteConfig.from_env()
    llm = build_llm(cfg)
    embedder = build_embedder(cfg)

    answer_cache = None
    if cfg.answer_cache:
        from ragnite.memory.semcache import AnswerCache

        answer_cache = AnswerCache(
            embedder=embedder,
            path=cfg.data_dir / "answer_cache",
            ttl_days=cfg.cache_ttl_days,
        )

    return RagEngine(
        store=build_store(cfg),
        embedder=embedder,
        llm=llm,
        reranker=build_reranker(cfg, llm),
        retrieval=cfg.retrieval,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        contextual=cfg.contextual,
        answer_cache=answer_cache,
    )


def build_memory(cfg: RagniteConfig | None = None) -> VectorMemory:
    """Legacy flat vector memory. Prefer :func:`build_memory_engine`."""
    cfg = cfg or RagniteConfig.from_env()
    return VectorMemory(embedder=build_embedder(cfg), path=cfg.data_dir / "memory")


def build_memory_engine(cfg: RagniteConfig | None = None) -> MemoryEngine:
    """Confidence-aware memory engine: typed memory bank + scorer + packer + semantic cache."""
    from ragnite.memory.bank import MemoryBank
    from ragnite.memory.engine import MemoryEngine
    from ragnite.memory.semcache import SemanticCache

    cfg = cfg or RagniteConfig.from_env()
    embedder = build_embedder(cfg)

    if cfg.store == "qdrant":
        from ragnite.store.qdrant import QdrantVectorStore

        bank_store: VectorStore = QdrantVectorStore(
            url=cfg.qdrant_url,
            collection=f"ragnite_memory_{cfg.collection}",
            api_key=_env("QDRANT_API_KEY") or None,
        )
        bank = MemoryBank(embedder=embedder, store=bank_store)
    else:
        bank = MemoryBank(embedder=embedder, path=cfg.data_dir / "memory_bank")

    cache = SemanticCache(
        embedder=embedder,
        path=cfg.data_dir / "semcache",
        threshold=cfg.cache_threshold,
        ttl_days=cfg.cache_ttl_days,
    )
    return MemoryEngine(bank=bank, cache=cache, default_budget_tokens=cfg.memory_budget_tokens)
