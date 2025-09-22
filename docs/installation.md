# Installation Guide

## Installing from PyPI

The recommended way to install Palabra AI SDK:

```bash
pip install palabra-ai
```

Or using uv (faster):
```bash
uv pip install palabra-ai
```

## Installing from Source

### For users
```bash
git clone https://github.com/PalabraAI/palabra-ai-python.git
cd palabra-ai-python
pip install .
```

### For developers
Install in editable mode to make changes:
```bash
git clone https://github.com/PalabraAI/palabra-ai-python.git
cd palabra-ai-python

# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e .
```

See [Contributing Guide](../CONTRIBUTING.md) for detailed development setup.

## Installing from GitHub Releases

Download the latest wheel file from [Releases](https://github.com/PalabraAI/palabra-ai-python/releases):

```bash
# Download the .whl file, then:
pip install palabra_ai-0.1.0-py3-none-any.whl
```

## Using with Docker

Pull the pre-built Docker image from GitHub Container Registry:

```bash
docker pull ghcr.io/palabraai/palabra-ai-sdk:latest

# Test the installation
docker run ghcr.io/palabraai/palabra-ai-sdk:latest

# Use as base image in your Dockerfile
FROM ghcr.io/palabraai/palabra-ai-sdk:latest
COPY your_app.py /app/
CMD ["python", "/app/your_app.py"]
```

## System Requirements

### Python Version
- Python 3.11 or higher

### Operating Systems
- Linux (Ubuntu 20.04+, Debian 11+, etc.)
- macOS 11+ (Big Sur and later)
- Windows 10/11

### Audio Dependencies
For microphone/speaker support, you may need:

**Linux:**
```bash
sudo apt-get install portaudio19-dev
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
- Audio drivers are usually included
- If issues occur, install [Visual C++ Redistributables](https://support.microsoft.com/en-us/help/2977003/)

## Verify Installation

Check that the package is installed correctly:

```python
import palabra_ai
print(palabra_ai.__version__)
# Output: 0.1.0
```

Test basic functionality:
```python
from palabra_ai import PalabraAI, EN, ES

# Should not raise any import errors
print("Installation successful!")
```

## Troubleshooting

### Import Error
If you get `ModuleNotFoundError: No module named 'palabra_ai'`:
- Make sure you activated your virtual environment
- Verify installation: `pip list | grep palabra-ai`
- Try reinstalling: `pip install --force-reinstall palabra-ai`

### Audio Issues
If you get audio-related errors:
- **Linux/macOS**: Install portaudio (see System Requirements)
- **All platforms**: Try `pip install --upgrade sounddevice`

### Version Conflicts
If you get dependency conflicts:
```bash
# Create a fresh virtual environment
python -m venv fresh_env
src fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install palabra-ai
```

## Next Steps

1. Set up your API credentials:
   ```bash
   export PALABRA_CLIENT_ID=your_key_here
   export PALABRA_CLIENT_SECRET=your_secret_here
   ```

2. Try the [Quick Start](../README.md#quick-start) example

3. Explore the [examples](https://github.com/PalabraAI/palabra-ai-python/tree/main/examples) directory

## Support

Need help with installation?
- Check [GitHub Issues](https://github.com/PalabraAI/palabra-ai-python/issues)
- Email: api.support@palabra.ai
