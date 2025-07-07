from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from livekit import rtc
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from palabra_ai.base.enum import Channel, Direction
from palabra_ai.base.message import Message
from palabra_ai.base.realtime_adapter import RealtimeAdapter
from palabra_ai.constant import WS_TIMEOUT
from palabra_ai.task.realtime import RtMsg
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug, error

# Import WebRTC components from the WebRTC adapter
from .realtime_webrtc import (
    _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX,
    _PALABRA_TRANSLATOR_TRACK_NAME_PREFIX,
    AudioPublication,
    AudioTrackSettings,
)


class RemoteAudioTrack:
    def __init__(
        self,
        tg: asyncio.TaskGroup,
        lang: str,
        participant: rtc.RemoteParticipant,
        publication: rtc.RemoteTrackPublication,
    ):
        self.tg = tg
        self.lang = lang
        self.participant = participant
        self.publication = publication
        self._listen_task = None

    def start_listening(self, q: asyncio.Queue[rtc.AudioFrame]):
        if not self._listen_task:
            self._listen_task = self.tg.create_task(self.listen(q), name="Mixed:listen")

    async def listen(self, q: asyncio.Queue[rtc.AudioFrame]):
        stream = rtc.AudioStream(self.publication.track)
        try:
            async for frame in stream:
                frame: rtc.AudioFrameEvent
                try:
                    await q.put(frame.frame)
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    debug(f"RemoteAudioTrack {self.lang} listen cancelled during put")
                    raise
        except asyncio.CancelledError:
            debug(f"Cancelled listening to {self.lang} track")
            raise
        finally:
            q.put_nowait(None)
            debug(f"Closing {self.lang} stream")
            try:
                await stream.aclose()
            except asyncio.CancelledError:
                debug(f"RemoteAudioTrack {self.lang} stream close cancelled")
            except Exception as e:
                error(f"Error closing {self.lang} stream: {e}")
            debug(f"Closed {self.lang} stream")
            self._listen_task = None

    async def stop_listening(self):
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass


def mark_received(q: asyncio.Queue):
    """Mark a message as received by completing the task"""
    try:
        q.task_done()
    except ValueError:
        pass


async def receive(q: asyncio.Queue, timeout: float = None) -> Any:
    """Receive a message from a queue with optional timeout"""
    try:
        if timeout:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        else:
            return await q.get()
    except asyncio.TimeoutError:
        return None


@dataclass
class MixedRealtimeAdapter(RealtimeAdapter):
    """Mixed WebSocket + WebRTC adapter (original behavior)"""
    
    # WebSocket components
    _websocket: Any = field(default=None, init=False)
    _keep_running: bool = field(default=True, init=False)
    _ws_connected: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _uri: str = field(default="", init=False)
    _ws_task: asyncio.Task | None = field(default=None, init=False)
    ws_raw_in_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    ws_out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    _raw_in_q: asyncio.Queue = field(default=None, init=False)
    
    # WebRTC components
    room: rtc.Room | None = field(default=None, init=False)
    room_out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    _publications: list[AudioPublication] = field(default_factory=list, init=False)
    
    # Mixed mode specific
    _audio_callbacks: dict[str, list[callable]] = field(default_factory=dict, init=False)
    _remote_tracks: dict[str, RemoteAudioTrack] = field(default_factory=dict, init=False)
    
    async def boot(self):
        """Initialize the adapter"""
        await self.connect()
    
    async def connect(self) -> None:
        """Connect to both WebSocket and WebRTC"""
        # Initialize WebSocket
        self._uri = f"{self.control_url}?token={self.jwt_token}"
        self._raw_in_q = self.ws_raw_in_foq.subscribe(self)
        self._ws_task = self.tg.create_task(self._ws_join(), name="Mixed:ws_join")
        
        # Initialize WebRTC
        self.room = rtc.Room()
        self._setup_webrtc_handlers()
        
        # Connect to WebRTC room
        options = rtc.RoomOptions(auto_subscribe=True)
        try:
            await self.room.connect(self.stream_url, self.jwt_token, options=options)
            debug(f"Connected to room {self.room.name}")
        except asyncio.CancelledError:
            debug("MixedRealtimeAdapter WebRTC connect cancelled")
            raise
        
        # Set up message routing
        self._setup_message_routing()
        debug("MixedRealtimeAdapter connected")
    
    async def _ws_join(self):
        """WebSocket connection loop with reconnection"""
        while self._keep_running:
            try:
                async with ws_connect(self._uri) as websocket:
                    self._websocket = websocket
                    self._ws_connected.set()
                    debug("WebSocket connected and ready")

                    receive_task = self.tg.create_task(
                        self._ws_receive_message(), name="Mixed:ws_receive"
                    )
                    send_task = self.tg.create_task(
                        self._ws_send_message(), name="Mixed:ws_send"
                    )

                    done, pending = await asyncio.wait(
                        [receive_task, send_task],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            debug("Task cancelled")
                            self._keep_running = False
            except asyncio.CancelledError:
                debug("MixedAdapter ws join cancelled")
                self._keep_running = False
                self._ws_connected.clear()
                raise
            except ConnectionClosed as e:
                if not self._keep_running:
                    debug(f"Connection closed during shutdown: {e}")
                else:
                    error(f"Connection closed with error: {e}")
            except Exception as e:
                error(f"Connection error: {e}")
            finally:
                if self._keep_running:
                    debug(f"Reconnecting to {self._uri}")
                    try:
                        await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        debug("MixedAdapter reconnect sleep cancelled")
                        self._keep_running = False
                        raise
                else:
                    debug("WebSocket shutting down gracefully")
                    self._ws_connected.clear()
                    break
    
    async def _ws_send_message(self):
        """Send messages through WebSocket"""
        debug("Starting _ws_send_message")
        while self._keep_running and self._websocket:
            try:
                try:
                    message = await asyncio.wait_for(
                        self._raw_in_q.get(), timeout=WS_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue
                await self._websocket.send(json.dumps(message))
                debug(f"Sent message: {message}")
                self._raw_in_q.task_done()
            except asyncio.CancelledError:
                debug("MixedAdapter _ws_send_message cancelled")
                raise
            except ConnectionClosed as e:
                if self._keep_running:
                    error(f"Unable to send message: {e}")
                break
            except Exception as e:
                error(f"Error in _ws_send_message: {e}")
                break
    
    async def _ws_receive_message(self):
        """Receive messages from WebSocket"""
        debug("Starting _ws_receive_message")
        while self._keep_running and self._websocket:
            try:
                debug("Waiting for WebSocket message...")
                async for raw_msg in self._websocket:
                    debug(f"Received message: {raw_msg}")
                    msg = Message.decode(raw_msg)
                    debug(f"Decoded message type: {type(msg).__name__}")
                    self.ws_out_foq.publish(msg)
                    
                    # Handle audio data messages if raw message contains them
                    # (audio comes as raw dict, not through Message types)
                    try:
                        if isinstance(raw_msg, (str, bytes)):
                            data = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                            if data.get("message_type") == "output_audio_data":
                                await self._handle_audio_output(data)
                    except:
                        pass
            except asyncio.CancelledError:
                debug("MixedAdapter _ws_receive_message cancelled")
                raise
            except ConnectionClosed as e:
                if self._keep_running:
                    error(f"Unable to receive message: {e}")
                break
            except Exception as e:
                error(f"Error in _ws_receive_message: {e}")
                break
    
    async def _handle_audio_output(self, msg: dict[str, Any]):
        """Handle audio output messages from WebSocket"""
        try:
            data = msg.get("data", {})
            language = data.get("language", "unknown")
            audio_data = base64.b64decode(data.get("data", ""))
            
            # Call registered callbacks for this language
            if language in self._audio_callbacks:
                for callback in self._audio_callbacks[language]:
                    try:
                        await callback(audio_data)
                    except Exception as e:
                        error(f"Error in audio callback: {e}")
        except Exception as e:
            error(f"Error handling audio output: {e}")
    
    def _setup_webrtc_handlers(self):
        """Set up WebRTC event handlers"""
        self.room.on("data_received", self._on_data_received)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
    
    def _on_data_received(self, packet: rtc.DataPacket):
        """Handle WebRTC data channel messages"""
        debug(f"Received data from {packet.participant.identity}: {packet.data}")
        msg = Message.decode(packet.data)
        self.room_out_foq.publish(msg)
    
    def _on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication,
                           participant: rtc.RemoteParticipant):
        """Handle track subscription"""
        debug(f"Track subscribed: {publication.sid}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            debug("Subscribed to an Audio Track")
    
    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connected"""
        debug(f"Participant connected: {participant.sid} {participant.identity}")
    
    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnected"""
        debug(f"Participant disconnected: {participant.sid} {participant.identity}")
    
    def _setup_message_routing(self):
        """Set up routing for all message channels"""
        # Route WebSocket IN messages
        ws_in_q = self.ws_raw_in_foq.subscribe(self, maxsize=0)
        self.tg.create_task(
            self._route_channel(Channel.WS, Direction.IN, ws_in_q),
            name="MixedAdapter:route_ws_in"
        )
        
        # Route WebSocket OUT messages
        ws_out_q = self.ws_out_foq.subscribe(self, maxsize=0)
        self.tg.create_task(
            self._route_channel(Channel.WS, Direction.OUT, ws_out_q),
            name="MixedAdapter:route_ws_out"
        )
        
        # Route WebRTC OUT messages
        webrtc_out_q = self.room_out_foq.subscribe(self, maxsize=0)
        self.tg.create_task(
            self._route_channel(Channel.WEBRTC, Direction.OUT, webrtc_out_q),
            name="MixedAdapter:route_webrtc_out"
        )
    
    async def _route_channel(self, channel: Channel, direction: Direction, 
                            from_q: asyncio.Queue):
        """Generic channel routing"""
        while not self.stopper:
            try:
                msg = await asyncio.wait_for(from_q.get(), timeout=1.0)
                if msg is None:
                    debug(f"Received None in {channel} {direction}, stopping route...")
                    break
                
                # Publish to appropriate queues
                if direction == Direction.IN:
                    self.in_foq.publish(RtMsg(channel, direction, msg))
                else:
                    debug(f"Publishing to out_foq: {channel} {direction} - {msg}")
                    self.out_foq.publish(RtMsg(channel, direction, msg))
                
                from_q.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                error(f"Error routing {channel} {direction}: {e}")
    
    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message through WebSocket"""
        if not self._keep_running:
            debug("MixedAdapter send called after shutdown")
            return
        try:
            self.ws_raw_in_foq.publish(message)
        except asyncio.CancelledError:
            debug("MixedAdapter send cancelled")
            raise
    
    async def set_translation_settings(self, settings: dict[str, Any]) -> None:
        """Set translation settings and create audio publication"""
        # Update settings for mixed mode:
        # - Both input and output through WebRTC (server requirement)
        if "input_stream" in settings:
            settings["input_stream"]["source"] = {
                "type": "webrtc"
            }
        if "output_stream" in settings:
            settings["output_stream"]["target"] = {
                "type": "webrtc"
            }
        
        # Wait for WebSocket to be connected before sending settings
        debug("Waiting for WebSocket connection...")
        await asyncio.wait_for(self._ws_connected.wait(), timeout=5.0)
        
        # Send settings through WebSocket
        debug(f"Sending translation settings: {settings}")
        await self.send_message({"message_type": "set_task", "data": settings})
        debug("Translation settings sent")
        
        # Create audio publication for WebRTC
        track_settings = AudioTrackSettings()
        publication = await AudioPublication.create(self.room, track_settings)
        
        try:
            # Wait for translator participant
            translator = await self._wait_for_participant_join(
                _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX, timeout=5
            )
            debug(f"Palabra translator participant joined: {translator.identity}")
        except TimeoutError:
            raise RuntimeError("Timeout. Palabra translator did not appear in the room")
        except asyncio.CancelledError:
            debug("MixedAdapter set_translation_settings cancelled")
            await publication.close()
            raise
        
        self._publications.append(publication)
    
    async def get_translation_settings(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """Get current translation settings"""
        start = time.perf_counter()
        subscriber_id = "MixedAdapter.get_translation_settings"
        
        try:
            out_q = self.ws_out_foq.subscribe(subscriber_id, 5)
            while True:
                try:
                    debug("MixedAdapter get_translation_settings sending request")
                    await self.send_message({"message_type": "get_task", "data": {}})
                except asyncio.CancelledError:
                    debug("MixedAdapter get_translation_settings send cancelled")
                    raise

                if timeout and time.perf_counter() - start > timeout:
                    raise TimeoutError("Timeout waiting for translation cfg")

                try:
                    message = await receive(out_q, 1)
                except asyncio.CancelledError:
                    debug("MixedAdapter get_translation_settings receive cancelled")
                    raise

                if message is None:
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError:
                        debug("MixedAdapter get_translation_settings sleep cancelled")
                        raise
                    continue

                if message["message_type"] == "current_task":
                    mark_received(out_q)
                    return message["data"]

                await asyncio.sleep(0)
        finally:
            self.ws_out_foq.unsubscribe(subscriber_id)
    
    async def get_translation_languages(self, timeout: Optional[int] = None) -> list[str]:
        """Get list of translation languages"""
        _get_trans_settings = self.get_translation_settings
        if timeout:
            _get_trans_settings = partial(_get_trans_settings, timeout=timeout)
        try:
            translation_settings = await _get_trans_settings()
        except asyncio.CancelledError:
            debug("MixedAdapter get_translation_languages cancelled")
            raise
        return [
            translation["target_language"]
            for translation in translation_settings["pipeline"]["translations"]
        ]
    
    async def get_translation_tracks(self, langs: list[str] | None = None) -> dict[str, RemoteAudioTrack]:
        """Get translation tracks for specified languages"""
        response = {}
        try:
            langs = langs or await self.get_translation_languages()
            participant = await self._wait_for_participant_join(
                _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX
            )
            for lang in langs:
                publication = await self._wait_for_track_publish(
                    participant, _PALABRA_TRANSLATOR_TRACK_NAME_PREFIX + lang
                )
                response[lang] = RemoteAudioTrack(
                    self.tg, lang, participant, publication
                )
        except asyncio.CancelledError:
            debug("MixedAdapter get_translation_tracks cancelled")
            raise
        return response
    
    async def _wait_for_participant_join(self, participant_identity: str, 
                                       timeout: int | float = None) -> rtc.RemoteParticipant:
        """Wait for a participant to join"""
        async def f():
            while True:
                for participant in self.room.remote_participants.values():
                    if (
                        str(participant.identity)
                        .lower()
                        .startswith(participant_identity.lower())
                    ):
                        return participant
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    debug("wait_for_participant_join sleep cancelled")
                    raise

        if timeout is None:
            return await f()
        try:
            return await asyncio.wait_for(f(), timeout=timeout)
        except asyncio.CancelledError:
            debug("wait_for_participant_join cancelled")
            raise
    
    async def _wait_for_track_publish(self, participant: rtc.RemoteParticipant, 
                                    name: str, timeout: int | float = None) -> rtc.RemoteTrackPublication:
        """Wait for a track to be published"""
        async def f():
            while True:
                for track in participant.track_publications.values():
                    if all(
                        [
                            str(track.name).lower().startswith(name.lower()),
                            track.kind == rtc.TrackKind.KIND_AUDIO,
                            track.track is not None,
                        ]
                    ):
                        return track
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    debug("wait_for_track_publish sleep cancelled")
                    raise

        if timeout is None:
            return await f()
        try:
            return await asyncio.wait_for(f(), timeout=timeout)
        except asyncio.CancelledError:
            debug("wait_for_track_publish cancelled")
            raise
    
    async def publish_audio(self, audio_data: bytes) -> None:
        """Publish audio through WebRTC"""
        # Find the first available publication
        if self._publications:
            await self._publications[0].push(audio_data)
    
    async def subscribe_to_audio(self, language: str, callback: callable) -> None:
        """Subscribe to translated audio for a specific language"""
        if language not in self._audio_callbacks:
            self._audio_callbacks[language] = []
        self._audio_callbacks[language].append(callback)
        
        # In mixed mode, translations come through WebSocket, not WebRTC tracks
        # so we just register the callback and wait for audio_output messages
    
    async def _process_track_audio(self, language: str, audio_queue: asyncio.Queue):
        """Process audio from a remote track"""
        while not self.stopper:
            try:
                frame = await audio_queue.get()
                if frame is None:
                    debug(f"Audio track for {language} ended")
                    break
                
                # Convert frame to bytes and call callbacks
                audio_data = bytes(frame.data)
                for callback in self._audio_callbacks.get(language, []):
                    try:
                        await callback(audio_data)
                    except Exception as e:
                        error(f"Error in audio callback: {e}")
            except Exception as e:
                error(f"Error processing {language} audio: {e}")
    
    async def do(self):
        """Main loop"""
        while not self.stopper:
            await asyncio.sleep(1.0)
    
    async def exit(self):
        """Clean up and close connections"""
        self.in_foq.publish(None)
        self.out_foq.publish(None)
        self.ws_raw_in_foq.publish(None)
        self.ws_out_foq.publish(None)
        self.room_out_foq.publish(None)
        
        # Stop all remote track listeners
        for track in self._remote_tracks.values():
            await track.stop_listening()
        
        await self.close()
    
    async def close(self) -> None:
        """Close all connections"""
        # Stop WebSocket
        if not self._keep_running:
            return

        self._keep_running = False

        try:
            await self.send_message({"message_type": "end_task", "data": {"force": True}})
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            debug("MixedAdapter close cancelled during send/wait")

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            try:
                await self._websocket.close()
            except asyncio.CancelledError:
                debug("MixedAdapter websocket close cancelled")
            except Exception as e:
                error(f"Error closing websocket: {e}")
        
        # Close WebRTC publications
        for publication in self._publications:
            try:
                await publication.close()
            except asyncio.CancelledError:
                debug("MixedAdapter publication close cancelled")
                continue
            except Exception as e:
                error(f"Error closing publication: {e}")

        # Disconnect from room
        if self.room:
            try:
                await self.room.disconnect()
            except asyncio.CancelledError:
                debug("MixedAdapter room disconnect cancelled")
                try:
                    await asyncio.wait_for(self.room.disconnect(), timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    error("MixedAdapter force disconnect failed")
            except Exception as e:
                error(f"Error disconnecting: {e}")