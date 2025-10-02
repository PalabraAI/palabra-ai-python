from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from aioshutdown import SIGHUP, SIGINT, SIGTERM

from palabra_ai.config import CLIENT_ID, CLIENT_SECRET, DEEP_DEBUG, Config
from palabra_ai.debug.hang_coroutines import diagnose_hanging_tasks
from palabra_ai.exc import ConfigurationError, unwrap_exceptions
from palabra_ai.internal.rest import PalabraRESTClient, SessionCredentials
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

    def run(
        self,
        cfg: Config,
        stopper: TaskEvent | None = None,
        no_raise=False,
        signal_handlers=False,
    ) -> RunResult | None:
        """
        Run the translation synchronously (blocking until completion).

        Args:
            cfg: Configuration for the translation
            stopper: Optional TaskEvent to control stopping
            no_raise: If True, return RunResult with exception instead of raising
            signal_handlers: If True, install Ctrl+C signal handlers (default: False)

        Returns:
            RunResult with execution status and logs, or None if interrupted
        """
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass

        try:
            if signal_handlers:
                # Run with signal handlers (old behavior)
                with SIGTERM | SIGHUP | SIGINT as shutdown_loop:
                    run_result = shutdown_loop.run_until_complete(
                        self.arun(cfg, stopper, no_raise)
                    )
                    return run_result
            else:
                # Run without signal handlers (new default)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    run_result = loop.run_until_complete(
                        self.arun(cfg, stopper, no_raise)
                    )
                    return run_result
                finally:
                    loop.close()
        except KeyboardInterrupt:
            debug("Received keyboard interrupt (Ctrl+C)")
            return None
        except BaseException as e:
            error(f"An error occurred during execution: {e}")
            if no_raise:
                return RunResult(ok=False, exc=e, eos=False)
            raise e
        finally:
            debug("Shutdown complete")

    async def arun(
        self,
        cfg: Config,
        stopper: TaskEvent | None = None,
        no_raise=False,
    ) -> RunResult:
        """
        Run the translation asynchronously (returns awaitable coroutine).

        Args:
            cfg: Configuration for the translation
            stopper: Optional TaskEvent to control stopping
            no_raise: If True, return RunResult with exception instead of raising

        Returns:
            RunResult with execution status and logs
            - If no_raise=False: returns RunResult(ok=True) or raises exception
            - If no_raise=True: always returns RunResult (ok=True or ok=False with exc)
        """

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
                        except (TimeoutError, asyncio.CancelledError):
                            debug(
                                "Logger task timeout or cancelled, checking result anyway"
                            )

                    # Try to get the result
                    log_data = manager.logger.result
                    if not log_data:
                        debug("Logger.result is None, trying to call exit() directly")
                        try:
                            log_data = await asyncio.wait_for(
                                manager.logger.exit(), timeout=2.0
                            )
                        except Exception as e:
                            debug(f"Failed to get log_data from logger.exit(): {e}")
            except Exception as e:
                error(f"Failed to retrieve log_data: {e}")

            # Check if EOS was received (only relevant for WS)
            eos_received = manager.io.eos_received if manager.io else False

            # Return result with whatever we managed to get
            if no_raise:
                return RunResult(
                    ok=ok,
                    exc=exc if not ok else None,
                    log_data=log_data,
                    io_data=manager.io.io_data if manager.io else None,
                    eos=eos_received,
                )
            elif ok:
                return RunResult(
                    ok=True,
                    exc=None,
                    log_data=log_data,
                    io_data=manager.io.io_data if manager.io else None,
                    eos=eos_received,
                )
            else:
                # no_raise=False and there was an error - raise it
                raise exc

        try:
            async with self.process(cfg, stopper) as manager:
                if DEEP_DEBUG:
                    debug(diagnose_hanging_tasks())
                coro = _run_with_result(manager)
                result = await coro
                if DEEP_DEBUG:
                    debug(diagnose_hanging_tasks())

                # Ensure result is not None
                if result is None:
                    # This should not happen, but just in case
                    if no_raise:
                        return RunResult(
                            ok=False,
                            exc=RuntimeError("Unexpected None result"),
                            eos=False,
                        )
                    raise RuntimeError("Unexpected None result from _run_with_result")

                return result

        except BaseException as e:
            error(f"Error in PalabraAI.arun(): {e}")
            # When no_raise=True, ALWAYS return RunResult with ok=False
            if no_raise:
                return RunResult(ok=False, exc=e, eos=False)
            # When no_raise=False, ALWAYS raise exception
            raise
        finally:
            if DEEP_DEBUG:
                debug(diagnose_hanging_tasks())

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

        try:
            async with asyncio.TaskGroup() as tg:
                manager = Manager(cfg, credentials, stopper=stopper)(tg)
                yield manager
            success("🎉🎉🎉 Translation completed 🎉🎉🎉")

        except* asyncio.CancelledError as eg:
            debug("TaskGroup received CancelledError")
            # Re-raise CancelledError
            raise eg.exceptions[0] from eg
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
                except TimeoutError:
                    error(f"Timeout deleting session {credentials.id}")
                except Exception as e:
                    error(f"Failed to delete session {credentials.id}: {e}")

            debug(diagnose_hanging_tasks())
