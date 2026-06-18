.PHONY: install lint format test check

install:
	uv sync --dev

lint:
	uv run ruff check src tests examples

format:
	uv run ruff format src tests examples
	uv run ruff check --fix src tests examples

test:
	uv run pytest tests -q

check: lint test
	uv run ruff format --check src tests examples
