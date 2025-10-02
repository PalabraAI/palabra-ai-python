from dataclasses import KW_ONLY, dataclass, field

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect

from palabra_ai.audio import AudioFrame
from palabra_ai.enum import Channel, Direction, Kind
from palabra_ai.message import Dbg, IoEvent
from palabra_ai.task.io.base import Io
from palabra_ai.util.logger import debug, trace
from palabra_ai.util.timing import get_perf_ts, get_utc_ts


@dataclass
class WsIo(Io):
    _: KW_ONLY
    ws: ClientConnection | None = field(default=None, init=False)
    _ws_cm: object | None = field(default=None, init=False)

    @property
    def dsn(self) -> str:
        return f"{self.credentials.ws_url}?token={self.credentials.jwt_token}"

    @property
    def channel(self) -> Channel:
        return Channel.WS

    async def send_message(self, msg_data: bytes) -> None:
        await self.ws.send(msg_data)

    async def send_frame(self, frame: AudioFrame, raw: bytes | None = None) -> None:
        if not raw:
            raw = frame.to_ws()
        debug(f"<- {frame} / {frame.dbg_delta=}")
        if self.global_start_perf_ts is None:
            self.init_global_start_ts()
            frame._dbg.dawn_ts = 0.0
        await self.ws.send(raw)

    def new_input_frame(self) -> AudioFrame:
        return AudioFrame.create(*self.cfg.mode.for_input_audio_frame)

    async def ws_receiver(self):
        from palabra_ai.message import EosMessage, Message

        try:
            async for raw_msg in self.ws:
                perf_ts = get_perf_ts()
                utc_ts = get_utc_ts()
                if self.stopper or raw_msg is None:
                    debug("Stopping ws_receiver due to stopper or None message")
                    raise EOFError("WebSocket connection closed or stopper triggered")
                trace(f"-> {raw_msg[:30]}")
                audio_frame = AudioFrame.from_ws(
                    raw_msg,
                    sample_rate=self.cfg.mode.output_sample_rate,
                    num_channels=self.cfg.mode.num_channels,
                    perf_ts=perf_ts,
                )
                if audio_frame:
                    debug(f"-> {audio_frame!r}")
                    if self.cfg.benchmark:
                        _dbg = Dbg(
                            Kind.AUDIO,
                            Channel.WS,
                            Direction.OUT,
                            idx=next(self._idx),
                            num=next(self._out_audio_num),
                            dur_s=audio_frame.duration,
                            perf_ts=perf_ts,
                            utc_ts=utc_ts,
                        )
                        _dbg.calc_dawn_ts(self.global_start_perf_ts)
                        audio_frame._dbg = _dbg
                        self.bench_audio_foq.publish(audio_frame)
                        self.io_events.append(IoEvent(_dbg, raw_msg))
                    self.writer.q.put_nowait(audio_frame)
                else:
                    _dbg = Dbg(
                        Kind.MESSAGE,
                        Channel.WS,
                        Direction.OUT,
                        idx=next(self._idx),
                        num=next(self._out_audio_num),
                    )
                    _dbg.calc_dawn_ts(self.global_start_perf_ts)
                    msg = Message.decode(raw_msg)
                    msg._dbg = _dbg
                    self.out_msg_foq.publish(msg)
                    self.io_events.append(IoEvent(_dbg, raw_msg))
                    debug(f"-> {msg!r}")
                    if isinstance(msg, EosMessage):
                        self.eos_received = True
                        raise EOFError(f"End of stream received: {msg}")

        except EOFError as e:
            +self.eof  # noqa
            self.eos_received = True
            debug(f"EOF!!! {e}")
        finally:
            self.writer.q.put_nowait(None)
            self.out_msg_foq.publish(None)

    async def boot(self):
        """Start WebSocket connection"""
        # Create context manager and enter it
        self._ws_cm = ws_connect(self.dsn)
        self.ws = await self._ws_cm.__aenter__()

        # Verify connection is ready
        await self.ws.ping()
        self.sub_tg.create_task(self.ws_receiver(), name="WsIo:receiver")
        self.sub_tg.create_task(self.in_msg_sender(), name="WsIo:in_msg_sender")
        await self.set_task()

    async def exit(self):
        """Clean up WebSocket connection"""
        if self._ws_cm and self.ws:
            await self._ws_cm.__aexit__(None, None, None)
        self.ws = None
