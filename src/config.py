"""
config.py — Centralised configuration for the Self-Healing RAG pipeline.

All tunables live here so the rest of the codebase never touches ``os.getenv``
directly. Values are read from the environment (and a local ``.env`` file if
present) exactly once and exposed through the ``settings`` singleton.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

try:  # Loading .env is a convenience, not a hard requirement (e.g. in CI).
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is always installed in practice
    pass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable view of every runtime setting the pipeline needs."""

    # ── LLM ────────────────────────────────────────────────────────────────
    # provider: openai | groq | google | anthropic
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    # Empty model = use the provider's sensible default (see nodes.DEFAULT_MODELS).
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_temperature: float = 0.0
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    # all-MiniLM-L6-v2 produces 384-dim vectors. Change if you swap models.
    embedding_dim: int = field(default_factory=lambda: _get_int("EMBEDDING_DIM", 384))

    # ── Vector store (Qdrant) ──────────────────────────────────────────────
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "documents")
    )

    # ── Reranking (Cohere) ─────────────────────────────────────────────────
    cohere_api_key: str | None = field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "rerank-english-v3.0")
    )

    # ── Retrieval / graph tuning ───────────────────────────────────────────
    retrieve_top_k: int = field(default_factory=lambda: _get_int("RETRIEVE_TOP_K", 10))
    rerank_top_n: int = field(default_factory=lambda: _get_int("RERANK_TOP_N", 4))
    max_retries: int = field(default_factory=lambda: _get_int("MAX_RETRIES", 2))

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 64))

    def require(self, *names: str) -> None:
        """Raise a clear error if any required secret is missing.

        Call this from entry points (API startup, eval runner) so failures are
        loud and early rather than a cryptic 401 deep inside a request.
        """
        missing = [n.upper() for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in the keys."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


# Convenience module-level handle: ``from src.config import settings``
settings = get_settings()
