from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from aioshutdown import SIGHUP, SIGINT, SIGTERM

from palabra_ai.base.task_event import TaskEvent
from palabra_ai.config import CLIENT_ID, CLIENT_SECRET, DEEP_DEBUG, Config
from palabra_ai.debug.hang_coroutines import diagnose_hanging_tasks
from palabra_ai.exc import ConfigurationError
from palabra_ai.internal.rest import PalabraRESTClient
from palabra_ai.task.manager import Manager
from palabra_ai.util.logger import debug, error, warning


@dataclass
class PalabraAI:
    client_id: str | None = field(default=CLIENT_ID)
    client_secret: str | None = field(default=CLIENT_SECRET)
    api_endpoint: str = "https://api.palabra.ai"

    def __post_init__(self):
        if not self.client_id:
            raise ConfigurationError("PALABRA_CLIENT_ID is not set")
        if not self.client_secret:
            raise ConfigurationError("PALABRA_CLIENT_SECRET is not set")

    def run(self, cfg: Config, stopper: TaskEvent | None = None) -> None:
        async def _run():
            try:
                async with self.process(cfg, stopper) as manager:
                    if DEEP_DEBUG:
                        debug(diagnose_hanging_tasks())
                    await manager.task
                    if DEEP_DEBUG:
                        debug(diagnose_hanging_tasks())
            except BaseException as e:
                error(f"Error in PalabraAI.run(): {e}")
                raise
            finally:
                if DEEP_DEBUG:
                    debug(diagnose_hanging_tasks())

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            task = loop.create_task(_run(), name="PalabraAI")

            def handle_interrupt(sig, frame):
                # task.cancel()
                +stopper  # noqa
                raise KeyboardInterrupt()

            old_handler = signal.signal(signal.SIGINT, handle_interrupt)
            try:
                return task
            finally:
                signal.signal(signal.SIGINT, old_handler)
        else:
            try:
                import uvloop

                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            except ImportError:
                pass

            try:
                with SIGTERM | SIGHUP | SIGINT as shutdown_loop:
                    shutdown_loop.run_until_complete(_run())
            except KeyboardInterrupt:
                debug("Received keyboard interrupt (Ctrl+C)")
                return
            except Exception as e:
                error(f"An error occurred during execution: {e}")
                raise
            finally:
                debug("Shutdown complete")

    @contextlib.asynccontextmanager
    async def process(
        self, cfg: Config, stopper: TaskEvent | None = None
    ) -> AsyncIterator[Manager]:
        warning("🤖 Connecting to Palabra.ai API...")
        if stopper is None:
            stopper = TaskEvent()

        credentials = await PalabraRESTClient(
            self.client_id,
            self.client_secret,
            base_url=self.api_endpoint,
        ).create_session()

        manager = None
        try:
            async with asyncio.TaskGroup() as tg:
                manager = Manager(cfg, credentials, stopper=stopper)(tg)
                yield manager

        except* asyncio.CancelledError:
            debug("TaskGroup received CancelledError")
        except* Exception as eg:
            for e in eg.exceptions:
                if not isinstance(e, asyncio.CancelledError):
                    error(f"Translation failed: {e}")
            raise
        finally:
            warning("🎉🎉🎉 Translation completed 🎉🎉🎉")
            debug(diagnose_hanging_tasks())
