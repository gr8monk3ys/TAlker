.PHONY: install run setup clean lint format test dev check all evaluate rebuild-index

# Installation
install:
	poetry install

install-dev:
	poetry install --with dev

# Running the application
run:
	poetry run streamlit run src/dashboard/Home.py

dev: install
	poetry run streamlit run src/dashboard/Home.py --server.runOnSave true

# Setup from scratch
setup:
	python -m pip install --upgrade pip
	python -m pip install poetry
	poetry install --with dev

# Code Quality
lint:
	poetry run pylint src/piazza_bot src/dashboard

format:
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

check:
	poetry run black --check src/ tests/
	poetry run ruff check src/ tests/
	poetry run mypy src/dashboard/llm.py src/dashboard/evaluation.py

# Testing
test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

test-verbose:
	poetry run pytest -v -s

# RAG Operations
evaluate:
	poetry run python -m src.dashboard.evaluation

rebuild-index:
	poetry run python -c "from src.dashboard.llm import LlmChain; chain = LlmChain(); chain.rebuild_index()"

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.coverage' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name 'htmlcov' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -exec rm -rf {} +

clean-db:
	rm -rf data/.chroma_db

# Full check before commit
all: format lint test
	@echo "All checks passed!"

# Help
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make run          - Run the Streamlit app"
	@echo "  make dev          - Run with hot reload"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo "  make check        - Check code quality"
	@echo "  make evaluate     - Run RAG evaluation"
	@echo "  make rebuild-index- Rebuild vector index"
	@echo "  make clean        - Clean cache files"
	@echo "  make clean-db     - Remove ChromaDB data"
	@echo "  make all          - Format, lint, and test"
