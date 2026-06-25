"""Tests for the FastAPI app (the graph is faked via monkeypatch)."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.graph import graph as graph_module

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_returns_answer(monkeypatch):
    def fake_answer_query(question):
        return {
            "answer": "Copenhagen.",
            "sources": ["denmark.txt"],
            "contexts": ["Copenhagen is the capital."],
            "grounded": True,
            "critique": "grounded",
            "critique_reason": "supported",
            "retries": 0,
        }

    monkeypatch.setattr(graph_module, "answer_query", fake_answer_query)

    resp = client.post("/query", json={"question": "Capital of Denmark?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Copenhagen."
    assert body["grounded"] is True
    assert body["sources"] == ["denmark.txt"]


def test_empty_question_is_422():
    # Pydantic min_length=1 rejects empty strings before the handler runs.
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_whitespace_question_is_400(monkeypatch):
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400


def test_query_surfaces_errors_as_500(monkeypatch):
    def boom(question):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(graph_module, "answer_query", boom)
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 500
    assert "qdrant down" in resp.json()["detail"]
