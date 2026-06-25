"""Tests for document loading and chunking."""
import pytest

from src.ingestion.loader import chunk_documents, load_documents


def test_loads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta", encoding="utf-8")
    (tmp_path / "ignore.bin").write_text("nope", encoding="utf-8")

    docs = load_documents(tmp_path)

    sources = {d.metadata["source"] for d in docs}
    assert sources == {"a.txt", "b.md"}


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does_not_exist")


def test_chunking_splits_long_text(tmp_path):
    pytest.importorskip("langchain_text_splitters")
    (tmp_path / "long.txt").write_text("word " * 1000, encoding="utf-8")

    docs = load_documents(tmp_path)
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "long.txt" for c in chunks)
