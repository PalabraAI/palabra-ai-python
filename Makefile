.PHONY: format check lint test help build bench clean

format:
	uv run ruff format ./src

check:
	uv run ruff check --fix ./src

lint: format check

test:
	uv run pytest

# Docker benchmark commands
docker-help:
	@echo "Palabra AI Benchmark Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make build              Build Docker image"
	@echo ""
	@echo "Usage:"
	@echo "  make bench -- [ARGS...] Run benchmark with arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make bench -- --help"
	@echo "  make bench -- --config kim_results/DNA.json --out kim_results/DNA_Oct8 kim_results/DNA.wav"
	@echo ""
	@echo "Note: Use '--' to separate make options from benchmark arguments"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              Remove containers (keeps .venv volume)"
	@echo "  make rebuild            Full rebuild (removes everything)"

build:
	docker-compose build

# Auto-build if image doesn't exist
bench:
	@docker images | grep -q palabra-ai-python-benchmark || $(MAKE) build
	docker-compose run --rm benchmark $(filter-out $@,$(MAKECMDGOALS))

# Catch-all target to prevent "No rule to make target" errors
%:
	@:

clean:
	docker-compose down
	docker-compose rm -f

rebuild:
	docker-compose down -v
	docker-compose build --no-cache
