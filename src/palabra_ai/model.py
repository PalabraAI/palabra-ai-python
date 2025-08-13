from __future__ import annotations
from __future__ import annotations

from typing import Optional
from typing import Optional

from pydantic import BaseModel
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


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
    exc: Optional[BaseException] = None
    log_data: Optional[LogData] = Field(default=None, repr=False)
