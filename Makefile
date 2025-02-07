.PHONY: install run setup clean lint format test dev

install:
	poetry install

run:
	poetry run streamlit run src/dashboard/Home.py

setup:
	python -m pip install --upgrade pip
	python -m pip install poetry
	poetry install

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.coverage' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +

lint:
	poetry run pylint src/piazza_bot src/dashboard

format:
	poetry run black src/

test:
	poetry run pytest

dev: install
	poetry run streamlit run src/dashboard/Home.py