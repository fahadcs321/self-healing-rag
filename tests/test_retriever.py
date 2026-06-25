"""Tests for the Qdrant retriever (with injected fakes — no network)."""

from types import SimpleNamespace

from src.retrieval.retriever import QdrantRetriever


class FakeEmbedder:
    def encode(self, text):
        return [0.1, 0.2, 0.3]


class FakeQdrant:
    def __init__(self, hits):
        self._hits = hits
        self.last_kwargs = None

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return self._hits


def _hit(text, source, score):
    return SimpleNamespace(payload={"text": text, "source": source}, score=score)


def test_search_maps_hits_to_documents():
    hits = [_hit("alpha", "a.txt", 0.9), _hit("beta", "b.txt", 0.8)]
    retriever = QdrantRetriever(embedder=FakeEmbedder(), client=FakeQdrant(hits))

    docs = retriever.search("question", top_k=2)

    assert [d.page_content for d in docs] == ["alpha", "beta"]
    assert docs[0].metadata["source"] == "a.txt"
    assert docs[0].metadata["score"] == 0.9


def test_search_passes_top_k_to_client():
    client = FakeQdrant([])
    retriever = QdrantRetriever(embedder=FakeEmbedder(), client=client)

    retriever.search("q", top_k=7)

    assert client.last_kwargs["limit"] == 7


class FakeQdrantQueryPoints:
    """Newer qdrant-client API: ``query_points`` returning an object with .points."""

    def __init__(self, hits):
        self._hits = hits
        self.last_kwargs = None

    def query_points(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(points=self._hits)


def test_uses_query_points_when_available():
    # Regression: qdrant-client >= 1.12 removed .search() in favour of query_points.
    hits = [_hit("gamma", "g.txt", 0.7)]
    client = FakeQdrantQueryPoints(hits)
    retriever = QdrantRetriever(embedder=FakeEmbedder(), client=client)

    docs = retriever.search("q", top_k=3)

    assert client.last_kwargs["limit"] == 3
    assert "query" in client.last_kwargs  # new API uses `query=`, not `query_vector=`
    assert docs[0].page_content == "gamma"


def test_missing_payload_yields_empty_content():
    hit = SimpleNamespace(payload=None, score=0.5)
    retriever = QdrantRetriever(embedder=FakeEmbedder(), client=FakeQdrant([hit]))

    docs = retriever.search("q")

    assert docs[0].page_content == ""
    assert docs[0].metadata["source"] == "unknown"
