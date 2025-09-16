import asyncio
import ssl
import sys
from typing import Any

import aiohttp
import certifi
from pydantic import BaseModel, Field

from palabra_ai.exc import ConfigurationError, InvalidCredentialsError
from palabra_ai.util.logger import debug, error, warning


class SessionCredentials(BaseModel):
    id: str = Field(..., description="session id")
    publisher: str = Field(..., description="publisher token")
    subscriber: list[str] = Field(..., description="subscriber token")
    webrtc_room_name: str = Field(..., description="livekit room name")
    webrtc_url: str = Field(..., description="livekit url")
    ws_url: str = Field(..., description="websocket management api url")

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if not self.jwt_token or not self.ws_url or not self.webrtc_url:
            raise ConfigurationError("Missing JWT token, ws URL, or webrtc URL")

    @property
    def jwt_token(self) -> str:
        if not self.publisher:
            raise ConfigurationError(
                f"Publisher token is missing or invalid, got: {self.publisher}"
            )
        return self.publisher

    @property
    def room_name(self) -> str:
        return self.webrtc_room_name

    @property
    def stream_url(self) -> str:
        if not self.webrtc_url:
            raise ConfigurationError("Stream URL is missing")
        return self.webrtc_url

    @property
    def control_url(self) -> str:
        if not self.ws_url:
            raise ConfigurationError("Control (ws) URL is missing")
        return self.ws_url


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

    async def create_session(self, subscriber_count: int = 0) -> SessionCredentials:
        """
        Create a new streaming session
        """
        session = None
        try:
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), connector=connector
            )

            response = await session.post(
                url=f"{self.base_url}/session-storage/session",
                json={
                    "data": {
                        "subscriber_count": subscriber_count,
                        "intent": "api",
                    }
                },
                headers={
                    "ClientID": self.client_id,
                    "ClientSecret": self.client_secret,
                },
            )

            # Check for invalid credentials (404 typically means invalid client_id/client_secret)
            if response.status == 404:
                raise InvalidCredentialsError(
                    "Invalid API credentials. Please check your PALABRA_CLIENT_ID and PALABRA_CLIENT_SECRET."
                )

            response.raise_for_status()
            body = await response.json()
            assert body["ok"] is True, "Request has failed"

            result = SessionCredentials.model_validate(body["data"])
            debug(f"Session {result.id} created")
            return result

        except asyncio.CancelledError:
            warning("PalabraRESTClient create_session cancelled")
            raise
        except aiohttp.ClientConnectorError as e:
            if "certificate verify failed" in str(e).lower():
                error(f"SSL Certificate Error: {e}")
                if sys.platform == "darwin":
                    error("On macOS, please run:")
                    error(
                        f"/Applications/Python\\ {sys.version_info.major}.{sys.version_info.minor}/Install\\ Certificates.command"
                    )
                    error("Or see the README for SSL setup instructions")
                else:
                    error("Please ensure SSL certificates are properly installed")
                    error("Try: pip install --upgrade certifi")
            raise
        except Exception as e:
            error(f"Error creating session: {e}")
            raise
        finally:
            if session:
                await session.close()

    async def delete_session(self, session_id: str) -> None:
        """
        Delete a streaming session
        """
        session = None
        try:
            # Create SSL context with certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5), connector=connector
            )

            response = await session.delete(
                url=f"{self.base_url}/session-storage/sessions/{session_id}",
                headers={
                    "ClientID": self.client_id,
                    "ClientSecret": self.client_secret,
                },
            )

            response.raise_for_status()
            debug(f"Session {session_id} deleted")

        except asyncio.CancelledError:
            warning("PalabraRESTClient delete_session cancelled")
            raise
        except Exception as e:
            error(f"Error deleting session {session_id}: {e}")
            # Don't re-raise the exception to prevent blocking shutdown
        finally:
            if session:
                await session.close()
