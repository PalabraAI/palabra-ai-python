FROM ghcr.io/astral-sh/uv:python3.11-bookworm

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy files needed for package build
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies (will be cached if deps don't change)
RUN uv sync --frozen

# The rest of the code will be mounted as volume which will override src/
# This allows git pull without rebuild

CMD ["uv", "run", "python", "-m", "palabra_ai.benchmark", "--help"]
