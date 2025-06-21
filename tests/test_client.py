import asyncio
import signal
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from palabra_ai.exc import ConfigurationError


class TestPalabraAI:
    def test_init_no_client_id(self):
        with patch.dict('os.environ', {'PALABRA_CLIENT_ID': '', 'PALABRA_CLIENT_SECRET': 'secret'}):
            # Remove cached modules
            if 'palabra_ai.config' in sys.modules:
                del sys.modules['palabra_ai.config']
            if 'palabra_ai.client' in sys.modules:
                del sys.modules['palabra_ai.client']

            from palabra_ai.client import PalabraAI

            with pytest.raises(ConfigurationError, match="PALABRA_CLIENT_ID is not set"):
                PalabraAI()

    def test_init_no_client_secret(self):
        with patch.dict('os.environ', {'PALABRA_CLIENT_ID': 'test_id', 'PALABRA_CLIENT_SECRET': ''}):
            # Remove cached modules
            if 'palabra_ai.config' in sys.modules:
                del sys.modules['palabra_ai.config']
            if 'palabra_ai.client' in sys.modules:
                del sys.modules['palabra_ai.client']

            from palabra_ai.client import PalabraAI

            with pytest.raises(ConfigurationError, match="PALABRA_CLIENT_SECRET is not set"):
                PalabraAI()

    def test_init_success(self):
        from palabra_ai.client import PalabraAI
        client = PalabraAI(client_id="test_id", client_secret="test_secret")
        assert client.api_endpoint == "https://api.palabra.ai"
