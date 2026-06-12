from ragnite.store.base import Filters, VectorStore, match_filters
from ragnite.store.native import NativeVectorStore

__all__ = ["Filters", "VectorStore", "match_filters", "NativeVectorStore", "QdrantVectorStore"]


def __getattr__(name: str):
    if name == "QdrantVectorStore":  # requires the optional qdrant-client extra
        from ragnite.store.qdrant import QdrantVectorStore

        return QdrantVectorStore
    raise AttributeError(name)
