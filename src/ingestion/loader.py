"""
loader.py — Load source documents and split them into chunks.

Supports ``.txt``, ``.md`` and ``.pdf`` files under a directory tree. Returns
LangChain ``Document`` objects with ``source`` metadata preserved so answers can
cite where they came from.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config import settings

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _load_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_pdf(path: Path) -> List[Document]:
    # Imported lazily so loading plain text never requires pypdf.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: List[Document] = []
    for page_num, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        if content.strip():
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": path.name, "page": page_num},
                )
            )
    return docs


def load_documents(source_dir: str | Path) -> List[Document]:
    """Recursively load every supported document under ``source_dir``."""
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    documents: List[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.extend(_load_text(path))

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def load_and_chunk(source_dir: str | Path) -> List[Document]:
    """Convenience: load then chunk in one call."""
    return chunk_documents(load_documents(source_dir))
