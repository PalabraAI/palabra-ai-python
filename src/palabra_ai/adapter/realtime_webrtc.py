from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from livekit import rtc

from palabra_ai.base.enum import Channel, Direction
from palabra_ai.base.message import Message
from palabra_ai.base.realtime_adapter import RealtimeAdapter
from palabra_ai.task.realtime import RtMsg
from palabra_ai.util.fanout_queue import FanoutQueue
from palabra_ai.util.logger import debug, error


_PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX = "palabra_translator_"
_PALABRA_TRANSLATOR_TRACK_NAME_PREFIX = "translation_"


class AudioTrackSettings:
    def __init__(
        self,
        sample_rate: int = 48_000,
        num_channels: int = 1,
        track_name: str | None = None,
        chunk_duration_ms: int = 10,
        track_source: rtc.TrackSource = rtc.TrackSource.SOURCE_MICROPHONE,
        track_options: rtc.TrackPublishOptions | None = None,
        dtx: bool = False,
    ):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.track_name = track_name or str(uuid.uuid4())
        self.chunk_duration_ms = chunk_duration_ms
        self.track_source = track_source
        self.track_options = (
            track_options
            if track_options is not None
            else rtc.TrackPublishOptions(dtx=dtx)
        )
        self.track_options.source = self.track_source

        self.audio_source = rtc.AudioSource(self.sample_rate, self.num_channels)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track(
            self.track_name, self.audio_source
        )

    @property
    def chunk_size(self) -> int:
        return int(self.sample_rate * (self.chunk_duration_ms / 1000))

    def new_frame(self) -> rtc.AudioFrame:
        return rtc.AudioFrame.create(
            self.sample_rate, self.num_channels, self.chunk_size
        )


class AudioPublication:
    def __init__(self, room: rtc.Room, track_settings: AudioTrackSettings):
        self.room = room
        self.track_settings = track_settings
        self.publication: rtc.LocalTrackPublication | None = None

    @classmethod
    async def create(
        cls, room: rtc.Room, track_settings: AudioTrackSettings
    ) -> "AudioPublication":
        self = cls(room, track_settings)
        try:
            self.publication = await room.local_participant.publish_track(
                self.track_settings.audio_track, self.track_settings.track_options
            )
            debug("Published track %s", self.publication.sid)
        except asyncio.CancelledError:
            debug("AudioPublication create cancelled")
            raise
        return self

    async def close(self) -> None:
        if self.publication:
            try:
                await self.room.local_participant.unpublish_track(
                    self.publication.track.sid
                )
                debug("Unpublished track %s", self.publication.sid)
            except asyncio.CancelledError:
                debug("AudioPublication close cancelled")
                try:
                    await asyncio.wait_for(
                        self.room.local_participant.unpublish_track(
                            self.publication.track.sid
                        ),
                        timeout=1.0,
                    )
                except (TimeoutError, asyncio.CancelledError):
                    error("AudioPublication force unpublish failed")
            except Exception as e:
                error(f"Error unpublishing track: {e}")
        self.publication = None

    async def push(self, audio_bytes: bytes) -> None:
        samples_per_channel = self.track_settings.chunk_size
        total_samples = len(audio_bytes) // 2
        audio_frame = self.track_settings.new_frame()
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

        try:
            for i in range(0, total_samples, samples_per_channel):
                if asyncio.get_running_loop().is_closed():
                    break
                frame_chunk = audio_bytes[i * 2 : (i + samples_per_channel) * 2]

                if len(frame_chunk) < samples_per_channel * 2:
                    padded_chunk = np.zeros(samples_per_channel, dtype=np.int16)
                    frame_chunk = np.frombuffer(frame_chunk, dtype=np.int16)
                    padded_chunk[: len(frame_chunk)] = frame_chunk
                else:
                    padded_chunk = np.frombuffer(frame_chunk, dtype=np.int16)

                np.copyto(audio_data, padded_chunk)

                await self.track_settings.audio_source.capture_frame(audio_frame)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            debug("AudioPublication push cancelled")
            raise


@dataclass
class WebRTCRealtimeAdapter(RealtimeAdapter):
    """WebRTC-only adapter using DataChannel for control messages"""
    
    room: rtc.Room | None = field(default=None, init=False)
    room_out_foq: FanoutQueue = field(default_factory=FanoutQueue, init=False)
    _audio_callbacks: dict[str, list[callable]] = field(default_factory=dict, init=False)
    _audio_publication: AudioPublication | None = field(default=None, init=False)
    _publications: list[AudioPublication] = field(default_factory=list, init=False)
    _remote_tracks: dict[str, rtc.RemoteTrackPublication] = field(default_factory=dict, init=False)
    
    async def boot(self):
        """Initialize the adapter"""
        await self.connect()
    
    async def connect(self) -> None:
        """Connect to WebRTC room"""
        # Create room with event handlers
        self.room = rtc.Room()
        self._setup_event_handlers()
        
        # Connect to room
        options = rtc.RoomOptions(auto_subscribe=True)
        debug(f"Connecting to WebRTC room at {self.stream_url}")
        try:
            await self.room.connect(self.stream_url, self.jwt_token, options=options)
            debug(f"Connected to room {self.room.name}")
        except asyncio.CancelledError:
            debug("WebRTCRealtimeAdapter connect cancelled")
            raise
        
        debug("WebRTCRealtimeAdapter connected")
    
    def _setup_event_handlers(self):
        """Set up WebRTC event handlers"""
        # Handle data channel messages
        self.room.on("data_received", self._on_data_received)
        
        # Handle track events
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_published", self._on_track_published)
        self.room.on("track_unpublished", self._on_track_unpublished)
        
        # Handle participant events
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        
        # Handle connection events
        self.room.on("connected", self._on_connected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("reconnecting", self._on_reconnecting)
        self.room.on("reconnected", self._on_reconnected)
        
        # Route room messages
        room_out_q = self.room_out_foq.subscribe(self, maxsize=0)
        self.tg.create_task(
            self._route_webrtc_out(room_out_q),
            name="WebRTCAdapter:route_out"
        )
    
    def _on_data_received(self, packet: rtc.DataPacket):
        """Handle data channel messages"""
        debug(f"Received data from {packet.participant.identity}: {packet.data}")
        try:
            msg = Message.decode(packet.data)
            # Publish to fanout queue and input queue
            self.room_out_foq.publish(msg)
            self.in_foq.publish(RtMsg(Channel.WEBRTC, Direction.IN, msg))
        except Exception as e:
            error(f"Error handling data packet: {e}")
    
    def _on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, 
                           participant: rtc.RemoteParticipant):
        """Handle track subscription"""
        debug(f"Track subscribed: {publication.sid}")
        if track.kind == rtc.TrackKind.KIND_AUDIO and "translation_" in publication.name:
            lang = publication.name.split("translation_")[-1]
            self._remote_tracks[lang] = publication
            
            # Start listening to the track
            self.tg.create_task(
                self._listen_to_track(track, lang),
                name=f"WebRTCAdapter:listen_{lang}"
            )
    
    def _on_track_published(self, publication: rtc.RemoteTrackPublication,
                          participant: rtc.RemoteParticipant):
        """Handle track published event"""
        debug(f"Track published: {publication.sid} from {participant.identity}")
    
    def _on_track_unpublished(self, publication: rtc.RemoteTrackPublication,
                            participant: rtc.RemoteParticipant):
        """Handle track unpublished event"""
        debug(f"Track unpublished: {publication.sid}")
    
    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connected"""
        debug(f"Participant connected: {participant.sid} {participant.identity}")
    
    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnected"""
        debug(f"Participant disconnected: {participant.sid} {participant.identity}")
    
    def _on_connected(self):
        """Handle connected event"""
        debug("WebRTC connected")
    
    def _on_disconnected(self):
        """Handle disconnected event"""
        debug("WebRTC disconnected")
    
    def _on_reconnecting(self):
        """Handle reconnecting event"""
        debug("WebRTC reconnecting")
    
    def _on_reconnected(self):
        """Handle reconnected event"""
        debug("WebRTC reconnected")
    
    async def _listen_to_track(self, track: rtc.Track, language: str):
        """Listen to audio track and call callbacks"""
        stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
        
        try:
            async for event in stream:
                frame: rtc.AudioFrameEvent = event
                audio_data = bytes(frame.frame.data)
                
                # Call registered callbacks
                if language in self._audio_callbacks:
                    for callback in self._audio_callbacks[language]:
                        try:
                            await callback(audio_data)
                        except Exception as e:
                            error(f"Error in audio callback: {e}")
        except asyncio.CancelledError:
            debug(f"Stopped listening to {language} track")
        finally:
            await stream.aclose()
    
    async def _route_webrtc_out(self, from_q: asyncio.Queue):
        """Route outgoing WebRTC messages"""
        while not self.stopper:
            try:
                msg = await asyncio.wait_for(from_q.get(), timeout=1.0)
                if msg is None:
                    debug("Received None in WebRTC OUT, stopping route...")
                    break
                
                self.out_foq.publish(RtMsg(Channel.WEBRTC, Direction.OUT, msg))
                from_q.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                error(f"Error routing WebRTC OUT: {e}")
    
    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message through DataChannel"""
        message_bytes = json.dumps(message).encode("utf-8")
        await self.room.local_participant.publish_data(message_bytes, reliable=True)
    
    async def set_translation_settings(self, settings: dict[str, Any]) -> None:
        """Set translation settings and create audio publication"""
        # Update settings to use WebRTC for input/output
        if "input_stream" in settings:
            settings["input_stream"]["source"] = {"type": "webrtc"}
        if "output_stream" in settings:
            settings["output_stream"]["target"] = {"type": "webrtc"}
        
        # Send settings through DataChannel
        await self.send_message({
            "message_type": "set_task",
            "data": settings
        })
        
        # Create audio publication for input
        track_settings = AudioTrackSettings()
        self._audio_publication = await AudioPublication.create(self.room, track_settings)
        self._publications.append(self._audio_publication)
        
        # Wait for translator participant
        try:
            await self._wait_for_participant(_PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX, timeout=5)
            debug("Translator participant joined")
        except TimeoutError:
            error("Timeout waiting for translator participant")
    
    async def get_translation_settings(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """Get current translation settings through DataChannel"""
        start = time.perf_counter()
        response_future = asyncio.Future()
        
        def handle_response(packet: rtc.DataPacket):
            try:
                msg = Message.decode(packet.data)
                if msg.get("message_type") == "current_task":
                    response_future.set_result(msg["data"])
            except:
                pass
        
        # Temporarily add handler
        self.room.on("data_received", handle_response)
        
        try:
            # Send request
            await self.send_message({"message_type": "get_task", "data": {}})
            
            # Wait for response
            if timeout:
                return await asyncio.wait_for(response_future, timeout=timeout)
            else:
                return await response_future
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout waiting for translation settings")
        finally:
            # Remove temporary handler
            self.room.off("data_received", handle_response)
    
    async def publish_audio(self, audio_data: bytes) -> None:
        """Publish audio data through WebRTC"""
        if self._audio_publication:
            await self._audio_publication.push(audio_data)
    
    async def subscribe_to_audio(self, language: str, callback: callable) -> None:
        """Subscribe to translated audio for a specific language"""
        if language not in self._audio_callbacks:
            self._audio_callbacks[language] = []
        self._audio_callbacks[language].append(callback)
        
        # Wait for translator participant if needed
        if language not in self._remote_tracks:
            await self._wait_for_translation_track(language)
    
    async def _wait_for_participant(self, identity_prefix: str, timeout: int = 10):
        """Wait for a participant with specific identity prefix"""
        start = time.time()
        
        while time.time() - start < timeout:
            for participant in self.room.remote_participants.values():
                if participant.identity.lower().startswith(identity_prefix.lower()):
                    return participant
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for participant {identity_prefix}")
    
    async def _wait_for_translation_track(self, language: str, timeout: int = 10):
        """Wait for a specific translation track to be published"""
        start = time.time()
        track_name = _PALABRA_TRANSLATOR_TRACK_NAME_PREFIX + language
        
        while time.time() - start < timeout:
            # Check if track already exists
            for participant in self.room.remote_participants.values():
                if participant.identity.startswith(_PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX):
                    for pub in participant.track_publications.values():
                        if pub.name == track_name and pub.track is not None:
                            self._remote_tracks[language] = pub
                            return
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for {language} translation track")
    
    async def do(self):
        """Main loop"""
        while not self.stopper:
            await asyncio.sleep(1.0)
    
    async def exit(self):
        """Clean up and close connections"""
        self.in_foq.publish(None)
        self.out_foq.publish(None)
        self.room_out_foq.publish(None)
        await self.close()
    
    async def close(self) -> None:
        """Close WebRTC connection"""
        # Close all publications
        for publication in self._publications:
            try:
                await publication.close()
            except Exception as e:
                error(f"Error closing publication: {e}")
        
        # Disconnect from room
        if self.room:
            try:
                await self.room.disconnect()
            except asyncio.CancelledError:
                debug("WebRTC disconnect cancelled")
                try:
                    await asyncio.wait_for(self.room.disconnect(), timeout=1.0)
                except (TimeoutError, asyncio.CancelledError):
                    error("WebRTC force disconnect failed")
            except Exception as e:
                error(f"Error disconnecting: {e}")