from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ok: bool
    exc: BaseException | None = None
    log_data: LogData | None = Field(default=None, repr=False)
