# Palabra AI Benchmarking Guide

Comprehensive benchmarking and performance analysis tool for Palabra AI translation pipelines.

## Overview

The benchmark module provides detailed metrics and analysis for translation quality, latency, and performance. It processes audio files through the full translation pipeline while collecting extensive telemetry data.

**Key Features:**
- ⚡ End-to-end latency measurements
- 📊 Detailed performance metrics
- 🎯 Translation quality analysis
- 💾 Complete trace data export
- 📝 Comprehensive JSON reports
- 🔊 Debug audio output (input + output comparison)

## Quick Start

### Three Ways to Run Benchmarks

#### 1. Using Docker (Recommended)
```bash
# One-time setup
cp .env.example .env  # Add your PALABRA_CLIENT_ID and PALABRA_CLIENT_SECRET
make build

# Run benchmark
make bench -- examples/speech/en.mp3 en es --out ./results
```

#### 2. Using uv (Direct)
```bash
uv run python -m palabra_ai.benchmark examples/speech/en.mp3 en es --out ./results
```

#### 3. Using Make (Local)
```bash
make bench -- examples/speech/en.mp3 en es --out ./results
```

**Note:** When using `make bench`, you must use `--` to separate make options from benchmark arguments.

## Usage

### Basic Syntax

```bash
# With language arguments
<command> <audio_file> <source_lang> <target_lang> [options]

# With config file
<command> <audio_file> --config <config.json> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `audio_file` | Yes | Path to input audio file (any format supported by FFmpeg) |
| `source_lang` | Conditional | Source language code (required without `--config`) |
| `target_lang` | Conditional | Target language code (required without `--config`) |
| `--config <file>` | No | Load full configuration from JSON file |
| `--out <dir>` | No | Output directory for results (default: console only) |

### Language Codes

Use standard language codes: `en`, `es`, `fr`, `de`, `ru`, `ja`, `zh`, etc.

See [main README](../README.md#supported-languages-) for complete language list.

## Examples

### Simple Language Pair
```bash
# English to Spanish
uv run python -m palabra_ai.benchmark examples/speech/en.mp3 en es --out ./bench_results

# Arabic to English
uv run python -m palabra_ai.benchmark examples/speech/ar.mp3 ar en --out ./bench_results
```

### Using Configuration File
```bash
# With custom config
uv run python -m palabra_ai.benchmark examples/speech/en.mp3 \
  --config examples/benchmark_config.json \
  --out ./bench_results
```

### Docker Examples
```bash
# Simple benchmark with Docker
make bench -- examples/speech/es.mp3 es en --out ./results

# With config file
make bench -- examples/speech/nbc.wav --config examples/benchmark_config.json --out ./results
```

### Console-Only Output
```bash
# Skip file output, print report to console
uv run python -m palabra_ai.benchmark examples/speech/en.mp3 en es
```

## Output Files

When `--out` is specified, the benchmark creates a timestamped set of files:

### Generated Files

| File | Description |
|------|-------------|
| `<timestamp>_bench_report.json` | Complete analysis report with metrics |
| `<timestamp>_bench_trace.json` | Full pipeline trace data |
| `<timestamp>_bench_config.json` | Configuration used for benchmark |
| `<timestamp>_bench_sysinfo.json` | System information and environment |
| `<timestamp>_bench.log` | Detailed debug logs |
| `<timestamp>_bench_in_<lang>.wav` | Input audio (preprocessed) |
| `<timestamp>_bench_out_<lang>.wav` | Output audio (translated) |
| `<timestamp>_bench_runresult_debug.json` | Runtime result data |

### Report Contents

The JSON report includes:
- **Latency metrics**: End-to-end, ASR, translation, TTS timings
- **Sentence analysis**: Per-sentence breakdown with timestamps
- **Quality metrics**: Translation validation, confidence scores
- **Performance data**: Processing speed, queue levels, tempo adjustments
- **System info**: Hardware, OS, Python version, library versions

## Configuration

### Using Config Files

Create a JSON config file with your pipeline settings:

```json
{
  "pipeline": {
    "transcription": {
      "source_language": "en",
      "segment_confirmation_silence_threshold": 0.7,
      "sentence_splitter": {
        "enabled": true
      }
    },
    "translations": [
      {
        "target_language": "es",
        "speech_generation": {
          "voice_cloning": false,
          "voice_id": "default_low"
        }
      }
    ],
    "translation_queue_configs": {
      "global": {
        "desired_queue_level_ms": 5000,
        "max_queue_level_ms": 20000,
        "auto_tempo": true
      }
    }
  }
}
```

Then run:
```bash
uv run python -m palabra_ai.benchmark audio.wav --config config.json --out ./results
```

### Environment Variables

Set credentials via environment variables:
```bash
export PALABRA_CLIENT_ID=your_client_id
export PALABRA_CLIENT_SECRET=your_client_secret
```

Or use `.env` file:
```bash
PALABRA_CLIENT_ID=your_client_id
PALABRA_CLIENT_SECRET=your_client_secret
```

## Benchmark Features

### Automatic Settings
- Forces 100ms chunk duration for optimal performance
- Enables all message types for complete data collection
- Captures full trace data and debug logs
- Generates progress bar with real-time status

### Progress Tracking
The benchmark shows real-time progress:
```
Processing en→es: 45%|████████████▌             | [00:23<00:28]
```

### Error Handling
- Saves partial results on errors
- Full traceback capture
- Debug audio export for analysis
- Detailed error logs

## Docker Workflow

### Setup (One Time)
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add credentials
# PALABRA_CLIENT_ID=...
# PALABRA_CLIENT_SECRET=...

# Build image
make build
```

### Run Benchmarks
```bash
# Update code (no rebuild needed)
git pull

# Run benchmark
make bench -- --config examples/benchmark_config.json --out ./results examples/speech/en.mp3

# Results saved to host filesystem
ls -la results/
```

### Docker Features
✅ **No rebuild on code changes** - Project directory is mounted as volume
✅ **Fast startup** - `.venv` is cached in Docker volume
✅ **File sharing** - Config files and results shared between host and container
✅ **Clean environment** - Isolated Python environment with all dependencies

## Troubleshooting

### Dependencies Changed?
Rebuild the Docker image:
```bash
make rebuild
```

### Permission Issues?
Container runs as your user, files should have correct permissions.

### Can't Find Audio Files?
Make sure paths are relative to project root (working directory is `/workspace` in container).

### Benchmark Fails?
Check the debug files in output directory:
- `*_bench_error.txt` - Error details and traceback
- `*_bench.log` - Full debug logs
- `*_bench_runresult_debug.json` - Runtime state

### Audio Processing Issues?
Ensure FFmpeg is installed and audio file is valid:
```bash
ffprobe your_audio.wav
```

### No Output Files?
Use `--out` flag to enable file output:
```bash
uv run python -m palabra_ai.benchmark audio.wav en es --out ./results
```

## Advanced Usage

### Analyzing Results

Parse the JSON report programmatically:
```python
import json
from pathlib import Path

report = json.loads(Path("results/20241010_150000_bench_report.json").read_text())

# Extract metrics
avg_latency = report["average_latency_ms"]
sentences = report["sentences"]

for s in sentences:
    print(f"Sentence {s['index']}: {s['latency_ms']}ms")
```

### Batch Processing

Run multiple benchmarks:
```bash
#!/bin/bash
for audio in examples/speech/*.wav; do
    uv run python -m palabra_ai.benchmark "$audio" en es --out "./results/$(basename $audio .wav)"
done
```

### Custom Analysis

The trace file (`*_bench_trace.json`) contains complete pipeline data:
- All messages exchanged
- Timing information
- Audio buffers
- Configuration state

## See Also

- [Main README](../README.md) - SDK documentation
- [Installation Guide](installation.md) - Setup instructions
- [Examples](../examples/) - Sample code and configs
