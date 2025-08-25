from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from aioshutdown import SIGHUP, SIGINT, SIGTERM

from palabra_ai.config import CLIENT_ID, CLIENT_SECRET, DEEP_DEBUG, Config
from palabra_ai.debug.hang_coroutines import diagnose_hanging_tasks
from palabra_ai.exc import ConfigurationError, unwrap_exceptions
from palabra_ai.internal.rest import PalabraRESTClient
from palabra_ai.internal.rest import SessionCredentials
from palabra_ai.model import RunResult
from palabra_ai.task.base import TaskEvent
from palabra_ai.task.manager import Manager
from palabra_ai.util.logger import debug, error, success


@dataclass
class PalabraAI:
    client_id: str | None = field(default=CLIENT_ID)
    client_secret: str | None = field(default=CLIENT_SECRET)
    api_endpoint: str = "https://api.palabra.ai"
    session_credentials: SessionCredentials | None = None

    def __post_init__(self):
        if not self.client_id:
            raise ConfigurationError("PALABRA_CLIENT_ID is not set")
        if not self.client_secret:
            raise ConfigurationError("PALABRA_CLIENT_SECRET is not set")

    def run(self, cfg: Config, stopper: TaskEvent | None = None, no_raise = False, without_signal_handlers = False) -> asyncio.Task | RunResult | None:
        async def _run() -> RunResult | None:
            async def _run_with_result(manager: Manager) -> RunResult:
                log_data = None
                exc = None
                ok = False

                try:
                    await manager.task
                    ok = True
                except asyncio.CancelledError as e:
                    debug("Manager task was cancelled")
                    exc = e
                except BaseException as e:
                    error(f"Error in manager task: {e}")
                    exc = e

                # CRITICAL: Always try to get log_data from logger
                try:
                    if manager.logger and manager.logger._task:
                        # Give logger time to complete if still running
                        if not manager.logger._task.done():
                            debug("Waiting for logger to complete...")
                            try:
                                await asyncio.wait_for(manager.logger._task, timeout=5.0)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                debug("Logger task timeout or cancelled, checking result anyway")

                        # Try to get the result
                        log_data = manager.logger.result
                        if not log_data:
                            debug("Logger.result is None, trying to call exit() directly")
                            try:
                                log_data = await asyncio.wait_for(manager.logger.exit(), timeout=2.0)
                            except Exception as e:
                                debug(f"Failed to get log_data from logger.exit(): {e}")
                except Exception as e:
                    error(f"Failed to retrieve log_data: {e}")

                # Return result with whatever we managed to get
                if no_raise or ok:
                    return RunResult(ok=ok, exc=exc if not ok else None, log_data=log_data)
                elif exc:
                    # Save log_data before raising exception
                    raise exc

            try:
                async with self.process(cfg, stopper) as manager:
                    if DEEP_DEBUG:
                        debug(diagnose_hanging_tasks())
                    coro = _run_with_result(manager)
                    result = await coro
                    if DEEP_DEBUG:
                        debug(diagnose_hanging_tasks())
                    return result
            except BaseException as e:
                error(f"Error in PalabraAI.run(): {e}")
                if no_raise:
                    return RunResult(ok=False, exc=e)
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
                if without_signal_handlers:
                    # Run without signal handlers for environments where they're not supported
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        run_result = loop.run_until_complete(_run())
                        return run_result
                    finally:
                        loop.close()
                else:
                    # Normal run with signal handlers
                    with SIGTERM | SIGHUP | SIGINT as shutdown_loop:
                        run_result = shutdown_loop.run_until_complete(_run())
                        return run_result
            except KeyboardInterrupt:
                debug("Received keyboard interrupt (Ctrl+C)")
                return
            except BaseException as e:
                error(f"An error occurred during execution: {e}")
                if no_raise:
                    return RunResult(ok=False, exc=e)
                raise e
            finally:
                debug("Shutdown complete")

    @contextlib.asynccontextmanager
    async def process(
        self, cfg: Config, stopper: TaskEvent | None = None
    ) -> AsyncIterator[Manager]:
        success(f"🤖 Connecting to Palabra.ai API with {cfg.mode}...")
        if stopper is None:
            stopper = TaskEvent()
        
        # Track if we created the session internally
        session_created_internally = False
        rest_client = None
        
        if self.session_credentials is not None:
            credentials = self.session_credentials
        else:
            rest_client = PalabraRESTClient(
                self.client_id,
                self.client_secret,
                base_url=self.api_endpoint,
            )
            credentials = await rest_client.create_session()
            session_created_internally = True

        manager = None
        try:
            async with asyncio.TaskGroup() as tg:
                manager = Manager(cfg, credentials, stopper=stopper)(tg)
                yield manager
            success("🎉🎉🎉 Translation completed 🎉🎉🎉")

        except* asyncio.CancelledError:
            debug("TaskGroup received CancelledError")
        except* Exception as eg:
            excs = unwrap_exceptions(eg)
            excs_wo_cancel = [
                e for e in excs if not isinstance(e, asyncio.CancelledError)
            ]
            for e in excs:
                error(f"Unhandled exception: {e}")
            if not excs_wo_cancel:
                raise excs[0] from eg
            raise excs_wo_cancel[0] from eg
        finally:
            # Clean up session if it was created internally
            if session_created_internally and rest_client and credentials:
                try:
                    await asyncio.wait_for(
                        rest_client.delete_session(credentials.id), timeout=5.0
                    )
                    success(f"Successfully deleted session {credentials.id}")
                except asyncio.TimeoutError:
                    error(f"Timeout deleting session {credentials.id}")
                except Exception as e:
                    error(f"Failed to delete session {credentials.id}: {e}")
            
            debug(diagnose_hanging_tasks())
