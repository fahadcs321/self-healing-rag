"""
conftest.py — Shared fixtures and lightweight fakes for the test suite.

The pipeline isolates heavy/networked dependencies (the LLM, the embedding model,
Qdrant, Cohere) behind small seams, so tests inject simple in-memory fakes and run
fast, offline and deterministically — no API keys, no Docker, no model downloads.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.documents import Document


class FakeLLM:
    """A chat-model stand-in that returns scripted responses in order.

    Each ``invoke`` returns an object with a ``.content`` attribute, matching the
    LangChain chat interface the nodes rely on.
    """

    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    def __init__(self, responses: list[str] | Callable[[str], str]) -> None:
        self._responses = responses
        self._calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str):  # noqa: D401 - mimics ChatOpenAI.invoke
        self.prompts.append(prompt)
        if callable(self._responses):
            content = self._responses(prompt)
        else:
            content = self._responses[min(self._calls, len(self._responses) - 1)]
        self._calls += 1
        return self._Msg(content)


def make_docs(*pairs: tuple[str, str]) -> list[Document]:
    """Build Documents from (text, source) pairs."""
    return [Document(page_content=t, metadata={"source": s}) for t, s in pairs]


@pytest.fixture
def docs() -> list[Document]:
    return make_docs(
        ("RAGAS measures faithfulness and answer relevancy.", "ragas.txt"),
        ("LangGraph enables cyclic, stateful graphs.", "langgraph.txt"),
    )
