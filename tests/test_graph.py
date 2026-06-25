"""End-to-end graph tests with faked retrieval and LLM — exercises the loop."""

import pytest

from src.graph import graph as graph_module
from src.graph import nodes
from src.graph.graph import answer_query
from tests.conftest import FakeLLM, make_docs


@pytest.fixture(autouse=True)
def fake_retrieval(monkeypatch):
    class FakeRetriever:
        def search(self, query, top_k=None):
            return make_docs(("Copenhagen is the capital of Denmark.", "denmark.txt"))

    class FakeReranker:
        def rerank(self, query, docs, top_n=None):
            return docs

    monkeypatch.setattr(nodes, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(nodes, "get_reranker", lambda: FakeReranker())
    # Compile a fresh graph per test so patched nodes are wired in cleanly.
    graph_module.get_graph.cache_clear()


def _use_llm(monkeypatch, responses):
    llm = FakeLLM(responses)
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    return llm


def test_grounded_answer_returns_immediately(monkeypatch):
    _use_llm(
        monkeypatch,
        [
            "Copenhagen.",
            '{"verdict": "grounded", "reason": "supported by context"}',
        ],
    )
    result = answer_query("What is the capital of Denmark?")

    assert result["grounded"] is True
    assert result["answer"] == "Copenhagen."
    assert result["retries"] == 0
    assert result["contexts"]  # real chunk text captured for evaluation


def test_self_heals_then_succeeds(monkeypatch):
    _use_llm(
        monkeypatch,
        [
            "Paris.",  # generate (wrong)
            '{"verdict": "hallucinated", "reason": "not in context"}',  # critique
            "capital city of Denmark",  # rewrite_query
            "Copenhagen.",  # generate (retry)
            '{"verdict": "grounded", "reason": "supported"}',  # critique
        ],
    )
    result = answer_query("What is the capital of Denmark?")

    assert result["grounded"] is True
    assert result["retries"] == 1
    assert result["answer"] == "Copenhagen."


def test_insufficient_context_refuses(monkeypatch):
    _use_llm(
        monkeypatch,
        [
            "I don't have enough information to answer that.",
            '{"verdict": "insufficient", "reason": "context lacks the answer"}',
        ],
    )
    result = answer_query("What is the GDP of Mars?")

    assert result["grounded"] is False
    assert "don't have enough" in result["answer"].lower()


def test_gives_up_after_max_retries(monkeypatch):
    # Always hallucinated: should rewrite up to the budget then refuse.
    responses = []
    for _ in range(5):
        responses.append("some answer")
        responses.append('{"verdict": "hallucinated", "reason": "ungrounded"}')
        responses.append("rewritten query")
    _use_llm(monkeypatch, responses)

    result = answer_query("An impossible question")

    assert result["grounded"] is False
    assert result["retries"] == 2  # settings.max_retries


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        answer_query("   ")
