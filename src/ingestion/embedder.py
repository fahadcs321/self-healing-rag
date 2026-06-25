"""
embedder.py — Turn text into dense vectors with sentence-transformers.

A thin wrapper that keeps the model name and vector dimension in one place and
loads the (heavy) model lazily on first use.
"""
from __future__ import annotations

from typing import Any, List

from src.config import settings


class Embedder:
    """Encode strings into fixed-size float vectors."""

    def __init__(self, model: Any = None) -> None:
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def encode(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Embed a batch of texts, returning a list of plain float lists."""
        vectors = self.model.encode(texts, show_progress_bar=show_progress)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return vectors

    def encode_one(self, text: str) -> List[float]:
        """Embed a single string."""
        return self.encode([text])[0]
