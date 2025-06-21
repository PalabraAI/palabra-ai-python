import asyncio
import uuid

import numpy as np
from livekit import rtc

from palabra_ai.base.message import Message
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
    def __init__(self, room_client: "RoomClient", track_settings: AudioTrackSettings):
        self.room_client = room_client
        self.track_settings = track_settings
        self.publication: rtc.LocalTrackPublication | None = None

    @classmethod
    async def create(
        cls, room_client: "RoomClient", track_settings: AudioTrackSettings
    ) -> "AudioPublication":
        self = cls(room_client, track_settings)
        try:
            self.publication = await self.room_client.local_participant.publish_track(
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
                await self.room_client.local_participant.unpublish_track(
                    self.publication.track.sid
                )
                debug("Unpublished track %s", self.publication.sid)
            except asyncio.CancelledError:
                debug("AudioPublication close cancelled")
                try:
                    await asyncio.wait_for(
                        self.room_client.local_participant.unpublish_track(
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


class RoomClient(rtc.Room):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        default_callbacks = [
            ("data_received", self.on_data_received),
            ("participant_connected", self.on_participant_connected),
            ("participant_disconnected", self.on_participant_disconnected),
            ("local_track_published", self.on_local_track_published),
            ("active_speakers_changed", self.on_active_speakers_changed),
            ("local_track_unpublished", self.on_local_track_unpublished),
            ("track_published", self.on_track_published),
            ("track_unpublished", self.on_track_unpublished),
            ("track_subscribed", self.on_track_subscribed),
            ("track_unsubscribed", self.on_track_unsubscribed),
            ("track_muted", self.on_track_muted),
            ("track_unmuted", self.on_track_unmuted),
            ("connection_quality_changed", self.on_connection_quality_changed),
            ("track_subscription_failed", self.on_track_subscription_failed),
            ("connection_state_changed", self.on_connection_state_changed),
            ("connected", self.on_connected),
            ("disconnected", self.on_disconnected),
            ("reconnecting", self.on_reconnecting),
            ("reconnected", self.on_reconnected),
        ]
        super().__init__(loop=loop)
        self.out_foq = FanoutQueue()
        self._publications: list[AudioPublication] = []
        for event, callback in default_callbacks:
            self.on(event, callback)

    async def connect(
        self, url: str, token: str, options: rtc.RoomOptions | None = None
    ):
        options = options if options is not None else rtc.RoomOptions()

        debug("connecting to %s", url)
        try:
            await super().connect(url, token, options=options)
            debug("connected to room %s", self.name)
        except asyncio.CancelledError:
            debug("RoomClient connect cancelled")
            raise

    async def close(self) -> None:
        for publication in self._publications:
            try:
                await publication.close()
            except asyncio.CancelledError:
                debug("RoomClient publication close cancelled")
                continue
            except Exception as e:
                error(f"Error closing publication: {e}")

        try:
            await self.disconnect()
        except asyncio.CancelledError:
            debug("RoomClient disconnect cancelled")
            try:
                await asyncio.wait_for(self.disconnect(), timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                error("RoomClient force disconnect failed")
        except Exception as e:
            error(f"Error disconnecting: {e}")

    async def new_publication(
        self, track_settings: AudioTrackSettings
    ) -> AudioPublication:
        publication = await AudioPublication.create(self, track_settings)
        try:
            translator_participant = await self.wait_for_participant_join(
                _PALABRA_TRANSLATOR_PARTICIPANT_IDENTITY_PREFIX, timeout=5
            )
            debug(
                "Palabra translator participant joined: %s",
                translator_participant.identity,
            )
        except TimeoutError:
            raise RuntimeError(
                "Timeout. Palabra translator did not appear in the room"
            ) from None
        except asyncio.CancelledError:
            debug("RoomClient new_publication cancelled")
            await publication.close()
            raise
        self._publications.append(publication)
        return publication

    async def wait_for_participant_join(
        self, participant_identity: str, timeout: int | float = None
    ) -> rtc.RemoteParticipant:
        async def f():
            while True:
                for participant in self.remote_participants.values():
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

    async def wait_for_track_publish(
        self, participant: rtc.RemoteParticipant, name: str, timeout: int | float = None
    ) -> rtc.RemoteTrackPublication:
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

    def on_data_received(self, data: rtc.DataPacket):
        debug("received data from %s: %s", data.participant.identity, data.data)
        # Publish to fanout queue
        debug(f"Received packet: {data}"[:100])
        msg = Message.decode(data.data)
        self.out_foq.publish(msg)

    def on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        debug(
            "track published: %s from participant %s (%s)",
            publication.sid,
            participant.sid,
            participant.identity,
        )

    def on_track_unpublished(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        debug("track unpublished: %s", publication.sid)

    def on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        debug("participant connected: %s %s", participant.sid, participant.identity)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        debug("participant disconnected: %s %s", participant.sid, participant.identity)

    def on_local_track_published(
        self,
        publication: rtc.LocalTrackPublication,
        track: rtc.LocalAudioTrack | rtc.LocalVideoTrack,
    ):
        debug("local track published: %s", publication.sid)

    def on_active_speakers_changed(self, speakers: list[rtc.Participant]):
        debug("active speakers changed: %s", speakers)

    def on_local_track_unpublished(self, publication: rtc.LocalTrackPublication):
        debug("local track unpublished: %s", publication.sid)

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        debug("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            _video_stream = rtc.VideoStream(track)
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            debug("Subscribed to an Audio Track")
            _audio_stream = rtc.AudioStream(track)

    def on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        debug("track unsubscribed: %s", publication.sid)

    def on_track_muted(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        debug("track muted: %s", publication.sid)

    def on_track_unmuted(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        debug("track unmuted: %s", publication.sid)

    def on_connection_quality_changed(
        self, participant: rtc.Participant, quality: rtc.ConnectionQuality
    ):
        debug("connection quality changed for %s", participant.identity)

    def on_track_subscription_failed(
        self, participant: rtc.RemoteParticipant, track_sid: str, error: str
    ):
        debug("track subscription failed: %s %s", participant.identity, error)

    def on_connection_state_changed(self, state: rtc.ConnectionState):
        debug("connection state changed: %s", state)

    def on_connected(self) -> None:
        debug("connected")

    def on_disconnected(self) -> None:
        debug("disconnected")

    def on_reconnecting(self) -> None:
        debug("reconnecting")

    def on_reconnected(self) -> None:
        debug("reconnected")
