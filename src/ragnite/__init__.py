"""Ragnite — production-grade Retrieval-Augmented Generation.

Quick start:

    from ragnite import RagEngine, NativeVectorStore

    engine = RagEngine(store=NativeVectorStore(".ragnite/collections/default"))
    await engine.ingest_path("docs/")
    results = await engine.search("how does billing work?")

Or build everything from environment variables:

    from ragnite import build_engine
    engine = build_engine()
"""

from ragnite.config import RagniteConfig, build_engine, build_memory
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # engine & config
    "RagEngine",
    "RagniteConfig",
    "build_engine",
    "build_memory",
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
    # memory
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
