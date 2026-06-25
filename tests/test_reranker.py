"""Tests for the Cohere reranker (with an injected fake client)."""

from types import SimpleNamespace

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


def test_top_n_capped_to_available_docs():
    docs = make_docs(("only", "a"))
    fake = FakeCohere()
    reranker = CohereReranker(client=fake)

    reranker.rerank("q", docs, top_n=10)

    assert fake.last_kwargs["top_n"] == 1
