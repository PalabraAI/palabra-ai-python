import asyncio
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

from palabra_ai.exc import ConfigurationError
from palabra_ai.util.logger import error, warning


class SessionCredentials(BaseModel):
    publisher: list[str] = Field(..., description="publisher token")
    subscriber: list[str] = Field(..., description="subscriber token")
    room_name: str = Field(..., description="livekit room name")
    stream_url: str = Field(..., description="livekit url")
    control_url: str = Field(..., description="websocket management api url")

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if not self.jwt_token or not self.control_url or not self.stream_url:
            raise ConfigurationError("Missing JWT token, control URL, or stream URL")

    @property
    def jwt_token(self) -> str:
        if not len(self.publisher) > 0:
            raise ConfigurationError(
                f"Publisher token is missing or invalid, got: {self.publisher}"
            )
        return self.publisher[0]

    @property
    def ws_url(self) -> str:
        if not self.control_url:
            raise ConfigurationError("Control (ws) URL is missing")
        return self.control_url

    @property
    def webrtc_url(self) -> str:
        if not self.stream_url:
            raise ConfigurationError("Stream URL is missing")
        return self.stream_url


class PalabraRESTClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        timeout: int = 5,
        base_url: str = "https://api.palabra.ai",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url
        self.timeout = timeout

    async def create_session(
        self, publisher_count: int = 1, subscriber_count: int = 0
    ) -> SessionCredentials:
        """
        Create a new streaming session
        """
        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

            response = await session.post(
                url=f"{self.base_url}/session-storage/sessions",
                json={
                    "data": {
                        "publisher_count": publisher_count,
                        "subscriber_count": subscriber_count,
                    }
                },
                headers={
                    "ClientID": self.client_id,
                    "ClientSecret": self.client_secret,
                },
            )

            response.raise_for_status()
            body = await response.json()
            assert body["ok"] is True, "Request has failed"

            return SessionCredentials.model_validate(body["data"])

        except asyncio.CancelledError:
            warning("PalabraRESTClient create_session cancelled")
            raise
        except Exception as e:
            error(f"Error creating session: {e}")
            raise
        finally:
            if session:
                await session.close()
