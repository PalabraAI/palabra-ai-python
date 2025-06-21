import asyncio

import aiohttp
from pydantic import BaseModel, Field

from palabra_ai.util.logger import error, warning


class SessionCredentials(BaseModel):
    publisher: list[str] = Field(..., description="publisher token")
    subscriber: list[str] = Field(..., description="subscriber token")
    room_name: str = Field(..., description="livekit room name")
    stream_url: str = Field(..., description="livekit url")
    control_url: str = Field(..., description="websocket management api url")


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
