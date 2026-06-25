.PHONY: help install install-dev qdrant ingest api ui eval gate test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies
	pip install -r requirements-dev.txt

qdrant:  ## Start Qdrant locally via Docker
	docker compose up -d qdrant

ingest:  ## Ingest documents from data/raw into Qdrant
	python -m src.ingestion.indexer --source data/raw --recreate

api:  ## Run the FastAPI server with auto-reload
	uvicorn src.api.main:app --reload

ui:  ## Run the Streamlit demo
	streamlit run app/streamlit_app.py

eval:  ## Run RAGAS evaluation over the golden dataset
	python -m src.evaluation.run_eval --output results/eval_results.json

gate:  ## Check the CI quality gate against the last eval
	python -m src.evaluation.threshold --results results/eval_results.json

test:  ## Run the unit test suite
	pytest

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Auto-format with ruff
	ruff format src tests

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.egg-info
