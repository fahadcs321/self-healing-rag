# 🔁 Self-Healing RAG Pipeline

> A production-grade RAG system that **detects its own hallucinations, rewrites its own
> queries, and refuses to answer when it can't ground a response** — with automated
> RAGAS quality gates blocking regressions in CI/CD.

[![Live Demo](https://img.shields.io/badge/live%20demo-streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://self-healing-rags.streamlit.app)
[![RAG Quality Gate](https://github.com/fahadcs321/self-healing-rag/actions/workflows/eval_ci.yml/badge.svg)](https://github.com/fahadcs321/self-healing-rag/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Built with LangGraph](https://img.shields.io/badge/built%20with-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)

**▶ Live demo: [self-healing-rags.streamlit.app](https://self-healing-rags.streamlit.app)**

Built as a **stateful, cyclic LangGraph agent** — not a linear chain.

---

## The Problem

Standard RAG pipelines are linear and brittle:

```
Query → Retrieve → Generate → Return
```

They **fail silently**. When the retrieved chunks are irrelevant, the LLM hallucinates
a confident, wrong answer rather than admitting it doesn't know. There's no feedback
loop, no self-correction, and no way to catch quality regressions before they ship.

## The Solution: a cyclic, self-correcting agent

```
Query → Retrieve → Rerank → Generate → Critique ─┬─ grounded     → Return answer
            ▲                                     ├─ hallucinated → Rewrite query → (retry, max 2)
            └──────────── Rewrite query ──────────┘
                                                  └─ insufficient → Return "I don't know" (honestly)
```

A dedicated **LLM-as-judge** grades every answer. If it's hallucinated, the agent
rewrites the query and retries. If the context genuinely lacks the answer, it says so
instead of fabricating. Every response is scored by **RAGAS**, and **GitHub Actions
blocks any merge** that regresses faithfulness, relevancy, or hallucination rate.

---

## Architecture

```
┌─────────────┐
│  USER QUERY │
└──────┬──────┘
┌──────▼──────┐
│  RETRIEVE   │  Dense semantic search over Qdrant (top-k chunks)
└──────┬──────┘
┌──────▼──────┐
│   RERANK    │  Cohere Rerank cross-encoder (keep top-n)
└──────┬──────┘
┌──────▼──────┐
│  GENERATE   │  LLM answers using ONLY the reranked context
└──────┬──────┘
┌──────▼──────┐
│   CRITIQUE  │  LLM-as-judge → grounded / hallucinated / insufficient
└──────┬──────┘
  ┌────┴───────────────┬──────────────────┐
grounded         hallucinated         insufficient
  │              (retry < 2)               │
  ▼                   │                    ▼
RETURN ANSWER   REWRITE QUERY        RETURN "I don't know"
                      │
                      ▼
                  RETRIEVE (loop)
```

Each node is a small, pure function of a shared `RAGState`, which makes every step
independently unit-testable. See [`src/graph/`](src/graph/).

---

## Tech Stack

| Layer          | Tool                          | Why                                        |
|----------------|-------------------------------|--------------------------------------------|
| Orchestration  | **LangGraph**                 | Stateful, cyclic graphs — the self-heal loop |
| Vector DB      | **Qdrant** (Docker)           | Fast vector search, free local setup       |
| Embeddings     | **sentence-transformers**     | `all-MiniLM-L6-v2` — 384-dim, CPU, free    |
| Reranking      | **Cohere Rerank**             | Cross-encoder; biggest precision lever     |
| LLM            | **Groq + Llama 3.3 70B**      | fast, free-tier, swappable via env var     |
| Evaluation     | **RAGAS**                     | Faithfulness, relevancy, recall, precision |
| CI/CD          | **GitHub Actions**            | Blocks merges on quality regression        |
| API            | **FastAPI**                   | `/query` + `/health`                       |
| UI             | **Streamlit**                 | Demo with full self-healing trace          |
| Observability  | **LangSmith**                 | Traces every node in the graph             |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/fahadcs321/self-healing-rag
cd self-healing-rag
pip install -r requirements.txt          # or: make install

# 2. Configure secrets
cp .env.example .env                       # fill in OPENAI_API_KEY, COHERE_API_KEY, ...

# 3. Start Qdrant
docker compose up -d qdrant                # or: make qdrant

# 4. Ingest the sample corpus (data/raw/) into Qdrant
make ingest                                # python -m src.ingestion.indexer --source data/raw --recreate

# 5a. Run the API …
make api                                   # uvicorn src.api.main:app --reload  → http://localhost:8000/docs

# 5b. … or the Streamlit demo
make ui                                    # streamlit run app/streamlit_app.py

# 6. Evaluate and check the quality gate
make eval                                  # RAGAS over data/golden_dataset.json
make gate                                  # exits non-zero if any metric regresses
```

Query the API directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the RAGAS faithfulness metric measure?"}'
```

```json
{
  "answer": "Faithfulness measures whether the claims in the answer are supported by the retrieved context.",
  "grounded": true,
  "critique": "grounded",
  "critique_reason": "Every claim is supported by the context.",
  "retries": 0,
  "sources": ["ragas_evaluation.txt"]
}
```

---

## RAGAS Metrics & Quality Gate

The CI gate ([`src/evaluation/threshold.py`](src/evaluation/threshold.py)) fails the
build — and blocks the merge — if any metric crosses its bound:

| Metric             | Measures                                          | Gate        |
|--------------------|---------------------------------------------------|-------------|
| Faithfulness       | Are the answer's claims grounded in context?      | `≥ 0.80`    |
| Answer Relevancy   | Does the answer address the question?             | `≥ 0.75`    |
| Context Recall     | Did we retrieve the right chunks?                 | `≥ 0.70`    |
| Context Precision  | Are the retrieved chunks actually relevant?       | `≥ 0.65`    |
| Hallucination Rate | Fraction of answers with faithfulness `< 0.5`     | `≤ 0.05`    |

> Run `make eval` to generate `results/eval_results.json` on your own corpus and keys.
> Thresholds live in one place and are easy to tune for your domain.

---

## Project Structure

```
self-healing-rag/
├── src/
│   ├── config.py              # Centralised settings (env-driven)
│   ├── ingestion/
│   │   ├── loader.py          # Load + chunk documents (txt/md/pdf)
│   │   ├── embedder.py        # sentence-transformers wrapper
│   │   └── indexer.py         # Load → embed → upsert into Qdrant
│   ├── retrieval/
│   │   ├── retriever.py       # Dense search over Qdrant
│   │   └── reranker.py        # Cohere Rerank wrapper
│   ├── graph/
│   │   ├── state.py           # RAGState TypedDict
│   │   ├── prompts.py         # Generation / critique / rewrite prompts
│   │   ├── nodes.py           # The 7 graph nodes (pure functions)
│   │   ├── edges.py           # Conditional routing (the self-heal logic)
│   │   └── graph.py           # Assemble + compile; answer_query()
│   ├── evaluation/
│   │   ├── ragas_eval.py      # RAGAS scorer
│   │   ├── run_eval.py        # Run pipeline over golden set, write results
│   │   └── threshold.py       # CI pass/fail gate
│   └── api/
│       ├── schemas.py         # Pydantic request/response models
│       └── main.py            # FastAPI app
├── app/streamlit_app.py       # Demo UI with self-healing trace
├── data/
│   ├── raw/                   # Sample corpus (RAG / LangGraph / Qdrant docs)
│   └── golden_dataset.json    # 25 grounded Q&A pairs for evaluation
├── tests/                     # 48+ unit tests (offline, fully mocked)
├── notebooks/01_exploration.ipynb
├── .github/workflows/eval_ci.yml
├── Dockerfile · docker-compose.yml · Makefile
├── requirements.txt · requirements-dev.txt · pyproject.toml
└── README.md
```

---

## Testing

The pipeline isolates every heavy/networked dependency (LLM, embeddings, Qdrant,
Cohere) behind small seams, so the test suite runs **offline, deterministically, and
in under a second** — no API keys, no Docker, no model downloads.

```bash
pip install -r requirements-dev.txt
pytest                # 48+ tests
```

Highlights: the full retrieve→rerank→generate→critique→**self-heal loop** is tested
end-to-end with a scripted fake LLM ([`tests/test_graph.py`](tests/test_graph.py)),
including the "hallucinate → rewrite → succeed" path and the "give up honestly after
max retries" path.

---

## Design Notes

- **Dense retrieval + cross-encoder reranking.** Retrieval is dense semantic search;
  Cohere Rerank then re-scores the candidates, which is the single biggest precision
  lever. True hybrid (dense + sparse BM25 fusion) is a natural next step — the
  retriever is structured to accommodate it.
- **Lazy, injectable dependencies.** Models and clients are created lazily and can be
  injected, which is what makes the code both import-cheap and testable.
- **Honest refusal over confident error.** When the critic can't verify an answer, the
  system returns "I don't know" rather than a plausible fabrication.

## Roadmap

- [ ] Hybrid dense + sparse (BM25) retrieval with reciprocal rank fusion
- [ ] Streaming responses over the API
- [ ] Per-query LangSmith trace links surfaced in the Streamlit UI
- [ ] Expand the golden dataset to 100+ pairs

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/) · [Qdrant](https://qdrant.tech/documentation/quickstart/) · [RAGAS](https://docs.ragas.io/) · [Cohere Rerank](https://cohere.com/rerank)
- [Self-RAG](https://arxiv.org/abs/2310.11511) · [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)

---

## Author

**Muhammad Fahad** · [GitHub](https://github.com/fahadcs321) · [LinkedIn](https://www.linkedin.com/in/muhammad-fahad-89a1b0358/)

Licensed under the [MIT License](LICENSE).
