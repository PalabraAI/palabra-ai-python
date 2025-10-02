import asyncio as aio
import uuid
from dataclasses import KW_ONLY, dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from livekit import rtc

from palabra_ai.audio import AudioFrame
from palabra_ai.constant import (
    SLEEP_INTERVAL_SHORT,
)
from palabra_ai.enum import Channel, Direction, Kind
from palabra_ai.message import (
    Dbg,
    EosMessage,
    Message,
)
from palabra_ai.task.io.base import Io
from palabra_ai.util.aio import shutdown
from palabra_ai.util.logger import debug, error
from palabra_ai.util.timing import get_perf_ts

PALABRA_PEER_PREFIX = "palabra_translator_"
PALABRA_TRACK_PREFIX = "translation_"

if TYPE_CHECKING:
    from palabra_ai.lang import Language


@dataclass
class WebrtcIo(Io):
    _: KW_ONLY
    in_track_name: str | None = None
    track_source: rtc.TrackSource = rtc.TrackSource.SOURCE_MICROPHONE
    in_track_options: rtc.TrackPublishOptions = field(
        default_factory=partial(rtc.TrackPublishOptions, dtx=False, red=False)
    )
    in_audio_source: rtc.AudioSource | None = None
    room: rtc.Room | None = None
    room_options: rtc.RoomOptions = field(default_factory=rtc.RoomOptions)
    loop: aio.AbstractEventLoop | None = None
    out_tracks: dict[str, rtc.RemoteAudioTrack] = field(
        default_factory=dict, init=False
    )
    out_track_publications: dict[str, rtc.RemoteTrackPublication] = field(
        default_factory=dict, init=False
    )
    in_track: rtc.LocalAudioTrack | None = None
    peer: rtc.RemoteParticipant | None = None

    def __post_init__(self):
        self.room = rtc.Room()

    async def peer_appears(self) -> rtc.RemoteParticipant:
        debug("Waiting for Palabra peer to appear...")
        name = PALABRA_PEER_PREFIX.lower()
        try:
            while True:
                for peer in self.room.remote_participants.values():
                    if str(peer.identity).lower().startswith(name):
                        debug(f"Found Palabra peer: {peer.identity}")
                        return peer
                await aio.sleep(SLEEP_INTERVAL_SHORT)
        except (TimeoutError, aio.CancelledError):
            debug(f"Didn't wait Palabra peer {name!r} to appear")
            raise

    async def track_appears(self, lang: "Language") -> rtc.RemoteTrackPublication:
        debug(f"Waiting for translation track for {lang!r} to appear...")
        name = f"{PALABRA_TRACK_PREFIX}{lang.code}".lower()
        try:
            while True:
                for tpub in self.peer.track_publications.values():
                    if all(
                        [
                            str(tpub.name).lower().startswith(name),
                            tpub.kind == rtc.TrackKind.KIND_AUDIO,
                            tpub.track is not None,
                        ]
                    ):
                        debug(f"Found translation track: {tpub.name}")
                        return tpub
                await aio.sleep(SLEEP_INTERVAL_SHORT)
        except (TimeoutError, aio.CancelledError):
            debug(f"Didn't wait track {name!r} to appear")
            raise

    async def out_audio(self, lang: "Language"):
        debug(f"Starting audio stream for {lang!r}...")
        stream = rtc.AudioStream(self.out_tracks[lang.code])
        try:
            async for frame_ev in stream:
                frame_ev: rtc.AudioFrameEvent
                perf_ts = get_perf_ts()
                audio_frame = AudioFrame.from_rtc(frame_ev.frame, perf_ts=perf_ts)
                self.writer.q.put_nowait(audio_frame)
                if self.cfg.benchmark:
                    audio_frame._dbg = Dbg(
                        Kind.AUDIO,
                        Channel.WEBRTC,
                        Direction.OUT,
                        idx=next(self._idx),
                        num=next(self._out_audio_num),
                        dur_s=audio_frame.duration,
                    )
                    audio_frame._dbg.calc_dawn_ts(self.global_start_perf_ts)
                    self.bench_audio_foq.publish(audio_frame)
                await aio.sleep(0)
                if self.stopper or self.eof:
                    debug(f"Stopping audio stream for {lang!r} due to stopper")
                    return
        finally:
            debug(f"Closing audio stream for {lang!r}...")
            self.writer.q.put_nowait(None)
            await shutdown(stream.aclose())
            debug(f"Closed audio stream for {lang!r}")

    def on_data_received(self, data: rtc.DataPacket):
        _dbg = Dbg(
            Kind.MESSAGE,
            Channel.WEBRTC,
            Direction.OUT,
            idx=next(self._idx),
            num=next(self._out_msg_num),
        )
        _dbg.calc_dawn_ts(self.global_start_perf_ts)
        debug(f"Received packet: {data}"[:100])
        msg = Message.decode(data.data)
        msg._dbg = _dbg
        self.out_msg_foq.publish(msg)
        if isinstance(msg, EosMessage):
            debug(f"End of stream received: {msg}")
            +self.eof  # noqa
            self.out_msg_foq.publish(None)
            self.writer.q.put_nowait(None)

    @property
    def channel(self) -> Channel:
        return Channel.WEBRTC

    async def send_message(self, msg_data: bytes) -> None:
        await self.room.local_participant.publish_data(msg_data, reliable=True)

    async def send_frame(self, frame: AudioFrame) -> None:
        self.init_global_start_ts()
        return await self.in_audio_source.capture_frame(frame.to_rtc())

    async def boot(self):
        debug(f"WebrtcIo.boot() STARTED for {self.name} id={id(self)}")
        await self.room.connect(
            self.credentials.webrtc_url, self.credentials.jwt_token, self.room_options
        )
        self.room.on("data_received", self.on_data_received)
        lang = self.cfg.targets[0].lang  # TODO: many langs
        self.peer = await self.peer_appears()
        debug(f"WebrtcIo.boot() creating in_msg_sender task for {self.name}")
        self.sub_tg.create_task(self.in_msg_sender(), name="Io:in_msg_sender")

        debug(f"WebrtcIo.boot() calling set_task() for {self.name}")
        try:
            await self.set_task()
            debug(f"WebrtcIo.boot() set_task() completed for {self.name}")
        except Exception as e:
            error(f"WebrtcIo.boot() set_task() FAILED: {e}")
            raise

        self.in_track_name = self.in_track_name or f"{uuid.uuid4()}_{lang.code}"
        # noinspection PyTypeChecker
        self.in_track_options.source = self.track_source
        self.in_audio_source = rtc.AudioSource(
            self.cfg.mode.input_sample_rate, self.cfg.mode.num_channels
        )
        self.in_track = rtc.LocalAudioTrack.create_audio_track(
            self.in_track_name, self.in_audio_source
        )
        await self.room.local_participant.publish_track(
            self.in_track, self.in_track_options
        )

        pub = await self.track_appears(lang)
        self.out_track_publications[lang.code] = pub
        self.out_tracks[lang.code] = pub.track
        self.sub_tg.create_task(self.out_audio(lang), name=f"Io:out_audio({lang!r})")

    async def exit(self):
        if self.in_track:
            await shutdown(
                self.room.local_participant.unpublish_track(self.in_track.sid)
            )
        if self.room:
            await shutdown(self.room.disconnect())
