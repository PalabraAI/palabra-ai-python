import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from palabra_ai.task.stat import Stat


class TestTaskStat:
    @pytest.mark.asyncio
    async def test_boot_and_exit(self):
        manager = MagicMock()
        stat = Stat(manager)

        await stat.boot()  # Should complete without error

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await stat.exit()

    def test_banner_property(self):
        manager = MagicMock()
        manager.tasks = [
            MagicMock(_state=["🚀", "🟢"]),
            MagicMock(_state=["🚀"]),
            MagicMock(_state=[])
        ]

        stat = Stat(manager)
        banner = stat._banner

        assert banner == "🟢🚀⭕"

    def test_show_banner(self):
        manager = MagicMock()
        stat = Stat(manager)
        stat.show_banner()  # Just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_banner_cancelled(self):
        manager = MagicMock()
        manager.tasks = []
        stat = Stat(manager)
        stat.sub_tg = MagicMock()

        async def cancel_soon():
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        with patch.object(stat, 'banner', side_effect=cancel_soon):
            task = MagicMock()
            stat.sub_tg.create_task.return_value = task

            result = stat.run_banner()
            assert result == task

    @pytest.mark.asyncio
    async def test_do_state_change(self):
        manager = MagicMock()
        manager.tasks = [
            MagicMock(_state=["🚀"]),
            MagicMock(_state=[])
        ]

        stat = Stat(manager)

        async def change_state():
            await asyncio.sleep(0.05)
            manager.tasks[0]._state.append("🟢")
            await asyncio.sleep(0.05)
            stat.stopper.set()

        with patch('palabra_ai.config.DEEP_DEBUG', False):
            await asyncio.gather(
                stat.do(),
                change_state(),
                return_exceptions=True
            )

    @pytest.mark.asyncio
    async def test_do_deep_debug(self):
        manager = MagicMock()
        manager.tasks = []
        manager.cfg = MagicMock()
        manager.cfg.deep_debug = True

        stat = Stat(manager)

        async def stop_soon():
            await asyncio.sleep(0.1)
            stat.stopper.set()

        with patch('palabra_ai.config.DEEP_DEBUG', True):
            await asyncio.gather(
                stat.do(),
                stop_soon(),
                return_exceptions=True
            )
