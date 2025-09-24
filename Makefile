.PHONY: format check lint test

format:
	uv run ruff format ./src

check:
	uv run ruff check --fix ./src

lint: format check

test:
	uv run pytest
