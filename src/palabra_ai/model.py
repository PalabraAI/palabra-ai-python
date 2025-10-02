from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from palabra_ai.message import IoEvent


class LogData(BaseModel):
    version: str
    sysinfo: dict
    messages: list[dict]
    start_ts: float
    cfg: dict
    log_file: str
    trace_file: str
    debug: bool
    logs: list[str]


class IoData(BaseModel):
    start_perf_ts: float
    start_utc_ts: float
    in_sr: int  # sr = sample rate
    out_sr: int  # sr = sample rate
    mode: str
    channels: int
    events: list[IoEvent]
    count_events: int


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ok: bool
    exc: BaseException | None = None
    log_data: LogData | None = Field(default=None, repr=False)
    io_data: IoData | None = Field(default=None, repr=False)
    eos: bool = False
