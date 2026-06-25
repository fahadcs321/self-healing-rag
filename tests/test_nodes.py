"""Tests for the graph nodes (LLM / retriever / reranker all faked)."""
import pytest

from src.graph import nodes
from src.graph.nodes import (
    _parse_critique,
    critique,
    generate,
    rerank,
    retrieve,
    return_answer,
    return_idk,
    rewrite_query,
)
from tests.conftest import FakeLLM, make_docs


@pytest.fixture
def patch_llm(monkeypatch):
    def _set(responses):
        llm = FakeLLM(responses)
        monkeypatch.setattr(nodes, "get_llm", lambda: llm)
        return llm

    return _set


def test_retrieve_uses_rewritten_query(monkeypatch):
    captured = {}

    class FakeRetriever:
        def search(self, query, top_k=None):
            captured["query"] = query
            return make_docs(("chunk", "s.txt"))

    monkeypatch.setattr(nodes, "get_retriever", lambda: FakeRetriever())
    out = retrieve({"query": "orig", "rewritten_query": "better"})

    assert captured["query"] == "better"
    assert len(out["retrieved_docs"]) == 1


def test_rerank_delegates(monkeypatch):
    class FakeReranker:
        def rerank(self, query, docs, top_n=None):
            return docs[:1]

    monkeypatch.setattr(nodes, "get_reranker", lambda: FakeReranker())
    docs = make_docs(("a", "x"), ("b", "y"))
    out = rerank({"query": "q", "retrieved_docs": docs})

    assert len(out["reranked_docs"]) == 1


def test_generate_collects_sources_and_contexts(patch_llm):
    patch_llm(["The answer."])
    docs = make_docs(("ctx one", "a.txt"), ("ctx two", "b.txt"))
    out = generate({"query": "q", "reranked_docs": docs})

    assert out["answer"] == "The answer."
    assert out["sources"] == ["a.txt", "b.txt"]
    assert out["contexts"] == ["ctx one", "ctx two"]


def test_critique_parses_verdict(patch_llm):
    patch_llm(['{"verdict": "grounded", "reason": "all supported"}'])
    out = critique({"query": "q", "answer": "a", "reranked_docs": make_docs(("c", "s"))})

    assert out["critique"] == "grounded"
    assert out["critique_reason"] == "all supported"


def test_rewrite_increments_retry(patch_llm):
    patch_llm(["a sharper query"])
    out = rewrite_query({"query": "q", "critique_reason": "too vague", "retry_count": 1})

    assert out["rewritten_query"] == "a sharper query"
    assert out["retry_count"] == 2


def test_return_answer_marks_grounded():
    out = return_answer({"answer": "final"})
    assert out["final_answer"] == "final"
    assert out["grounded"] is True


def test_return_idk_is_honest():
    out = return_idk({"critique_reason": "no context"})
    assert out["grounded"] is False
    assert "no context" in out["final_answer"]


# ── _parse_critique robustness ────────────────────────────────────────────────

def test_parse_plain_json():
    assert _parse_critique('{"verdict": "hallucinated", "reason": "x"}')[0] == "hallucinated"


def test_parse_markdown_fenced_json():
    raw = '```json\n{"verdict": "grounded", "reason": "ok"}\n```'
    verdict, reason = _parse_critique(raw)
    assert verdict == "grounded"
    assert reason == "ok"


def test_parse_garbage_defaults_to_hallucinated():
    verdict, reason = _parse_critique("not json at all")
    assert verdict == "hallucinated"
    assert "unparseable" in reason


def test_parse_unknown_verdict_is_hallucinated():
    verdict, _ = _parse_critique('{"verdict": "maybe", "reason": "?"}')
    assert verdict == "hallucinated"
