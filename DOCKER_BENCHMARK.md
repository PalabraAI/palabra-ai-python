# Docker Benchmark Setup

Simple Docker setup for running palabra_ai benchmarks without installing dependencies locally.

## Quick Start

### 1. Setup (one time)

```bash
# Copy .env template and fill in your credentials
cp .env.example .env
# Edit .env and add your PALABRA_CLIENT_ID and PALABRA_CLIENT_SECRET

# Build Docker image
make build
# or: docker-compose build
```

### 2. Run benchmark

```bash
# Using Makefile (recommended)
make bench -- --config kim_results/DNA.json --out kim_results/DNA_Oct8 kim_results/DNA.wav

# Or directly with docker-compose
docker-compose run --rm benchmark --config kim_results/DNA.json --out kim_results/DNA_Oct8 kim_results/DNA.wav
```

**Note:** When using `make bench`, you must use `--` to separate make options from benchmark arguments.

## Features

✅ **No rebuild on code changes** - Project directory is mounted as volume, just `git pull` and run
✅ **Fast startup** - `.venv` is cached in Docker volume
✅ **File sharing** - Config files and results are shared between host and container
✅ **Clean environment** - Isolated Python environment with all dependencies

## Workflow

```bash
# Update code (no rebuild needed)
git pull

# Run benchmark
make bench -- --config path/to/config.json --out ./results audio.wav

# Results are saved to host filesystem
ls -la results/
```

## Cleanup

```bash
# Remove Docker volumes and containers
make clean
```

## Manual Commands

```bash
# Build image
docker-compose build

# Run benchmark
docker-compose run --rm benchmark [BENCHMARK_ARGS]

# Example
docker-compose run --rm benchmark --help
```

## Troubleshooting

**Dependencies changed?**
Rebuild the image:
```bash
make build
```

**Permission issues?**
Container runs as your user, files should have correct permissions.

**Can't find audio files?**
Make sure paths are relative to project root (working directory is `/workspace` in container).
