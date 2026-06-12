from ragnite.embed.base import EmbeddingProvider
from ragnite.embed.cache import EmbeddingCache
from ragnite.embed.fake import FakeEmbedder
from ragnite.embed.openai_compat import OpenAICompatEmbedder
from ragnite.embed.voyage import VoyageEmbedder

__all__ = [
    "EmbeddingProvider",
    "EmbeddingCache",
    "FakeEmbedder",
    "OpenAICompatEmbedder",
    "VoyageEmbedder",
    "LocalEmbedder",
]


def __getattr__(name: str):
    if name == "LocalEmbedder":  # requires the optional sentence-transformers extra
        from ragnite.embed.local import LocalEmbedder

        return LocalEmbedder
    raise AttributeError(name)
