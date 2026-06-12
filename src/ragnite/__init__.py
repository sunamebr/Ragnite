"""Ragnite — confidence-aware RAG memory engine for LLMs and coding agents.

Memory + conviction (the core product):

    from ragnite import build_memory_engine

    memory = build_memory_engine()
    await memory.remember_decision("We use Postgres 16 with pgbouncer.", subject="database")
    answer = await memory.recall("which database do we use?")
    answer.mode        # "direct" | "cautious" | "ask_clarification" | "search_more" | "refuse_guess"
    answer.confidence  # 0..1
    answer.context     # smallest token-budgeted evidence pack for the LLM

Document RAG (grounded answers with citations):

    from ragnite import build_engine

    engine = build_engine()
    await engine.ingest_path("docs/")
    result = await engine.ask("how does billing work?")
"""

from ragnite.config import RagniteConfig, build_engine, build_memory, build_memory_engine
from ragnite.embed import (
    EmbeddingCache,
    EmbeddingProvider,
    FakeEmbedder,
    OpenAICompatEmbedder,
    VoyageEmbedder,
)
from ragnite.errors import (
    ConfigError,
    GenerationError,
    IngestionError,
    MissingDependencyError,
    RagniteError,
    RetrievalError,
)
from ragnite.ingest import load_path, load_text
from ragnite.llm import ChatModel, LLMResponse, OpenAICompatChat
from ragnite.memory import (
    AnswerMode,
    CodeIndexStats,
    CodeMemory,
    ConfidencePolicy,
    ConfidenceReport,
    ConfidenceScorer,
    ConfidenceSignals,
    ContextPacker,
    Evidence,
    MemoryAnswer,
    MemoryBank,
    MemoryEngine,
    MemoryKind,
    MemoryRecord,
    PackedContext,
    SemanticCache,
)
from ragnite.rag import ConversationMemory, RagEngine, VectorMemory
from ragnite.retrieve import (
    BM25Index,
    CohereReranker,
    LLMReranker,
    RetrievalConfig,
    Retriever,
    VoyageReranker,
    rrf_fuse,
)
from ragnite.store import NativeVectorStore, VectorStore
from ragnite.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    IngestStats,
    ScoredChunk,
    StreamEvent,
    Usage,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # engines & config
    "MemoryEngine",
    "RagEngine",
    "RagniteConfig",
    "build_engine",
    "build_memory_engine",
    "build_memory",
    # memory subsystem
    "MemoryBank",
    "MemoryKind",
    "MemoryRecord",
    "MemoryAnswer",
    "Evidence",
    "AnswerMode",
    "ConfidencePolicy",
    "ConfidenceScorer",
    "ConfidenceReport",
    "ConfidenceSignals",
    "ContextPacker",
    "PackedContext",
    "SemanticCache",
    "CodeMemory",
    "CodeIndexStats",
    # types
    "Document",
    "Chunk",
    "ScoredChunk",
    "Answer",
    "Citation",
    "StreamEvent",
    "IngestStats",
    "Usage",
    # ingestion
    "load_path",
    "load_text",
    # embeddings
    "EmbeddingProvider",
    "EmbeddingCache",
    "VoyageEmbedder",
    "OpenAICompatEmbedder",
    "FakeEmbedder",
    # stores
    "VectorStore",
    "NativeVectorStore",
    # retrieval
    "Retriever",
    "RetrievalConfig",
    "BM25Index",
    "rrf_fuse",
    "CohereReranker",
    "VoyageReranker",
    "LLMReranker",
    # llm
    "ChatModel",
    "LLMResponse",
    "OpenAICompatChat",
    # legacy memory
    "VectorMemory",
    "ConversationMemory",
    # errors
    "RagniteError",
    "ConfigError",
    "MissingDependencyError",
    "IngestionError",
    "RetrievalError",
    "GenerationError",
]
