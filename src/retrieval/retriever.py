"""
retriever.py — Dense semantic search over Qdrant.

Wraps the embedding model and the Qdrant client behind a small, testable class.
Heavy dependencies (``sentence-transformers``, ``qdrant-client``) are imported
lazily inside ``__init__`` so importing this module is cheap and side-effect free
— unit tests can construct a retriever with injected fakes instead.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, List

from langchain_core.documents import Document

from src.config import settings


class QdrantRetriever:
    """Embed a query and fetch the most similar chunks from Qdrant.

    Parameters allow dependency injection for testing; in production they default
    to the configured embedding model and Qdrant instance.
    """

    def __init__(self, embedder: Any = None, client: Any = None) -> None:
        self._embedder = embedder
        self._client = client
        self.collection = settings.collection

    # ── Lazy resources ─────────────────────────────────────────────────────
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int | None = None) -> List[Document]:
        """Return up to ``top_k`` chunks ranked by cosine similarity."""
        top_k = top_k or settings.retrieve_top_k

        vector = self.embedder.encode(query)
        # SentenceTransformer returns a numpy array; Qdrant wants a plain list.
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        hits = self._query(vector, top_k)
        return [self._hit_to_document(h) for h in hits]

    def _query(self, vector: List[float], top_k: int) -> List[Any]:
        """Run the vector search, supporting both new and old qdrant-client APIs.

        qdrant-client >= 1.12 replaced ``search`` with ``query_points`` (which
        returns a response object whose ``.points`` holds the hits). Older
        versions still expose ``search`` returning the hits directly.
        """
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            return response.points
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

    @staticmethod
    def _hit_to_document(hit: Any) -> Document:
        payload = getattr(hit, "payload", None) or {}
        return Document(
            page_content=payload.get("text", ""),
            metadata={
                "source": payload.get("source", "unknown"),
                "page": payload.get("page"),
                "score": getattr(hit, "score", None),
            },
        )


@lru_cache(maxsize=1)
def get_retriever() -> QdrantRetriever:
    """Process-wide retriever singleton (loads the embedding model once)."""
    return QdrantRetriever()
