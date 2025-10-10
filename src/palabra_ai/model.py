from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from palabra_ai.message import IoEvent


class IoData(BaseModel):
    model_config = {"use_enum_values": True}
    start_perf_ts: float
    start_utc_ts: float
    in_sr: int  # sr = sample rate
    out_sr: int  # sr = sample rate
    mode: str
    channels: int
    events: list[IoEvent]
    count_events: int
    reader_x_title: str
    writer_x_title: str


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ok: bool
    exc: BaseException | None = None
    io_data: IoData | None = Field(default=None, repr=False)
    eos: bool = False
