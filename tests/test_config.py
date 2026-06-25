"""Tests for the configuration layer."""
import pytest

from src.config import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.embedding_dim == 384
    assert s.rerank_top_n <= s.retrieve_top_k
    assert s.max_retries >= 0


def test_singleton_is_cached():
    assert get_settings() is get_settings()


def test_require_raises_for_missing_keys(monkeypatch):
    s = Settings()
    object.__setattr__(s, "openai_api_key", None)
    with pytest.raises(RuntimeError) as exc:
        s.require("openai_api_key")
    assert "OPENAI_API_KEY" in str(exc.value)


def test_require_passes_when_present():
    s = Settings()
    object.__setattr__(s, "openai_api_key", "sk-test")
    s.require("openai_api_key")  # should not raise
