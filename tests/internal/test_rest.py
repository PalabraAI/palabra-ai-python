import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from palabra_ai.internal.rest import PalabraRESTClient


class TestPalabraRESTClient:
    @pytest.mark.asyncio
    async def test_create_session(self, mock_aiohttp):
        client = PalabraRESTClient("id", "secret")
        creds = await client.create_session()

        assert creds.publisher == ["token1"]
        assert creds.room_name == "test"
        assert creds.stream_url == "wss://test"

        mock_aiohttp.ClientSession.assert_called()

    @pytest.mark.asyncio
    async def test_create_session_error(self, mock_aiohttp):
        mock_aiohttp.ClientSession.return_value.post.side_effect = Exception("Network error")

        client = PalabraRESTClient("id", "secret")

        with pytest.raises(Exception) as exc_info:
            await client.create_session()

        assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_session_cancelled(self, mock_aiohttp):
        mock_aiohttp.ClientSession.return_value.post = AsyncMock(side_effect=asyncio.CancelledError)

        client = PalabraRESTClient("id", "secret")

        with pytest.raises(asyncio.CancelledError):
            await client.create_session()

    @pytest.mark.asyncio
    async def test_create_session_not_ok(self, mock_aiohttp):
        resp = AsyncMock()
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={"ok": False})

        mock_aiohttp.ClientSession.return_value.post = AsyncMock(return_value=resp)

        client = PalabraRESTClient("id", "secret")

        with pytest.raises(AssertionError) as exc_info:
            await client.create_session()

        assert "Request has failed" in str(exc_info.value)
