"""Tests for the Cohere reranker (with an injected fake client)."""

from types import SimpleNamespace

import pytest

from src.retrieval import reranker as reranker_mod
from src.retrieval.reranker import CohereReranker
from tests.conftest import make_docs


class FakeCohere:
    """Returns results that reverse the input order, top_n honoured."""

    def __init__(self):
        self.last_kwargs = None

    def rerank(self, model, query, documents, top_n):
        self.last_kwargs = {"model": model, "query": query, "top_n": top_n}
        order = list(range(len(documents)))[::-1][:top_n]
        results = [
            SimpleNamespace(index=i, relevance_score=1.0 - pos * 0.1) for pos, i in enumerate(order)
        ]
        return SimpleNamespace(results=results)


def test_rerank_reorders_and_trims():
    docs = make_docs(("first", "a"), ("second", "b"), ("third", "c"))
    reranker = CohereReranker(client=FakeCohere())

    ranked = reranker.rerank("q", docs, top_n=2)

    assert [d.page_content for d in ranked] == ["third", "second"]
    assert "rerank_score" in ranked[0].metadata


def test_rerank_empty_returns_empty():
    reranker = CohereReranker(client=FakeCohere())
    assert reranker.rerank("q", [], top_n=4) == []


class _RateLimitError(Exception):
    status_code = 429


class FlakyCohere:
    """Raises a 429 for the first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def rerank(self, model, query, documents, top_n):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _RateLimitError("rate limited")
        results = [SimpleNamespace(index=0, relevance_score=0.9)]
        return SimpleNamespace(results=results)


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(reranker_mod.time, "sleep", lambda _s: None)  # no real waiting
    fake = FlakyCohere(fail_times=2)
    reranker = CohereReranker(client=fake)

    ranked = reranker.rerank("q", make_docs(("only", "a")), top_n=1)

    assert fake.calls == 3  # 2 failures + 1 success
    assert ranked[0].page_content == "only"


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(reranker_mod.time, "sleep", lambda _s: None)
    fake = FlakyCohere(fail_times=999)
    reranker = CohereReranker(client=fake)

    with pytest.raises(_RateLimitError):
        reranker.rerank("q", make_docs(("only", "a")), top_n=1)


def test_top_n_capped_to_available_docs():
    docs = make_docs(("only", "a"))
    fake = FakeCohere()
    reranker = CohereReranker(client=fake)

    reranker.rerank("q", docs, top_n=10)

    assert fake.last_kwargs["top_n"] == 1
