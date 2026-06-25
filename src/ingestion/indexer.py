"""
indexer.py — Build the Qdrant index from a directory of documents.

Pipeline: load + chunk (loader.py) → embed (embedder.py) → upsert into Qdrant.

Usage:
    python -m src.ingestion.indexer --source data/raw --collection documents
    python -m src.ingestion.indexer --source data/raw --recreate
"""
from __future__ import annotations

import argparse
import uuid
from typing import Any, List

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_and_chunk


def _get_client() -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(client: Any, name: str, dim: int, recreate: bool = False) -> None:
    """Create the collection if it does not exist (or recreate it)."""
    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}

    if recreate and name in existing:
        client.delete_collection(name)
        existing.discard(name)

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{name}' (dim={dim}, cosine).")
    else:
        print(f"Collection '{name}' already exists — upserting into it.")


def _to_points(chunks: List[Document], vectors: List[List[float]]) -> List[Any]:
    from qdrant_client.models import PointStruct

    return [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def index(
    source_dir: str,
    collection: str | None = None,
    recreate: bool = False,
    batch_size: int = 128,
) -> int:
    """Ingest ``source_dir`` into Qdrant. Returns the number of chunks indexed."""
    collection = collection or settings.collection
    client = _get_client()
    embedder = Embedder()

    chunks = load_and_chunk(source_dir)
    if not chunks:
        print(f"No supported documents found under '{source_dir}'. Nothing to index.")
        return 0
    print(f"Loaded and chunked {len(chunks)} chunks from '{source_dir}'.")

    ensure_collection(client, collection, embedder.dim, recreate=recreate)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.encode([c.page_content for c in batch], show_progress=True)
        client.upsert(collection_name=collection, points=_to_points(batch, vectors))
        total += len(batch)
        print(f"  upserted {total}/{len(chunks)} chunks")

    print(f"Done. Indexed {total} chunks into '{collection}'.")
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--source", default="data/raw", help="Directory of documents.")
    parser.add_argument("--collection", default=settings.collection)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before indexing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    index(args.source, collection=args.collection, recreate=args.recreate)
