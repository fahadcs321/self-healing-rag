"""
reranker.py — Cohere Rerank wrapper.

Reranking is the single biggest precision lever in this pipeline: it re-scores the
``top_k`` candidates from dense retrieval against the query with a cross-encoder and
keeps only the best ``top_n``. The Cohere client is imported lazily so this module
imports without the SDK installed (handy for tests).
"""

from __future__ import annotations

import logging
import random
import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger("self_healing_rag.reranker")

# Cohere trial keys are capped at 10 requests/minute, so back off and retry on 429.
_MAX_RETRIES = 6
_BASE_DELAY = 6.0  # seconds; the trial limit resets per minute


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "TooManyRequests" in type(exc).__name__


class CohereReranker:
    """Re-order documents by relevance to a query using Cohere Rerank."""

    def __init__(self, client: Any = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import cohere

            self._client = cohere.Client(settings.cohere_api_key)
        return self._client

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Return the ``top_n`` most relevant documents, best first.

        Falls back to the original order (truncated) if there is nothing to rank.
        """
        top_n = top_n or settings.rerank_top_n
        if not documents:
            return []

        texts = [d.page_content for d in documents]
        response = self._rerank_with_retry(query, texts, min(top_n, len(texts)))

        ranked: list[Document] = []
        for result in response.results:
            doc = documents[result.index]
            # Attach the rerank score without mutating the caller's metadata.
            metadata = {**doc.metadata, "rerank_score": result.relevance_score}
            ranked.append(Document(page_content=doc.page_content, metadata=metadata))
        return ranked

    def _rerank_with_retry(self, query: str, texts: list[str], top_n: int) -> Any:
        """Call Cohere Rerank, backing off and retrying on 429 rate limits."""
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.rerank(
                    model=settings.rerank_model,
                    query=query,
                    documents=texts,
                    top_n=top_n,
                )
            except Exception as exc:  # noqa: BLE001
                if not _is_rate_limit(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BASE_DELAY * (attempt + 1) + random.uniform(0, 1)
                logger.warning(
                    "Cohere rate limit hit; retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(delay)
        raise RuntimeError("Cohere rerank failed after retries")  # pragma: no cover


@lru_cache(maxsize=1)
def get_reranker() -> CohereReranker:
    """Process-wide reranker singleton."""
    return CohereReranker()
