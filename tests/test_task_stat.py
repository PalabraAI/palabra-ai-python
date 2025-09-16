import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from palabra_ai.task.stat import Stat
from palabra_ai.task.base import TaskEvent
from palabra_ai.constant import SLEEP_INTERVAL_DEFAULT, SLEEP_INTERVAL_LONG, SLEEP_INTERVAL_MEDIUM

class TestStat:
    """Test Stat task"""

    @pytest.fixture
    def mock_manager(self):
        """Create mock manager"""
        manager = MagicMock()
        manager.cfg = MagicMock()
        manager.cfg.deep_debug = False
        manager.tasks = []
        return manager

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        return MagicMock()

    def test_init(self, mock_config, mock_manager):
        """Test Stat initialization"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        assert stat.cfg == mock_config
        assert stat.manager == mock_manager

    @pytest.mark.asyncio
    async def test_boot(self, mock_config, mock_manager):
        """Test boot method"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        # Boot should do nothing
        await stat.boot()

    @pytest.mark.asyncio
    async def test_do_normal_operation(self, mock_config, mock_manager):
        """Test do method normal operation"""
        stat = Stat(cfg=mock_config, manager=mock_manager)
        stat.stopper = TaskEvent()

        # Stop after a short delay
        async def stop_after_delay():
            await asyncio.sleep(0.01)
            stat.stopper.set()

        asyncio.create_task(stop_after_delay())

        with patch('palabra_ai.task.stat.debug') as mock_debug:
            await stat.do()

            # Should have logged stats
            mock_debug.assert_called()

    @pytest.mark.asyncio
    async def test_do_state_change(self, mock_config, mock_manager):
        """Test do method with state change"""
        stat = Stat(cfg=mock_config, manager=mock_manager)
        stat.stopper = TaskEvent()

        # Create changing state
        states = ["state1", "state2"]
        state_index = 0

        def get_state():
            nonlocal state_index
            result = states[state_index % len(states)]
            state_index += 1
            return result

        # Stop after some iterations
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stat.stopper.set()

        asyncio.create_task(stop_after_delay())

        with patch.object(Stat, 'stat_palabra_tasks', new_callable=PropertyMock) as mock_prop:
            mock_prop.side_effect = get_state

            with patch('palabra_ai.task.stat.debug') as mock_debug:
                await stat.do()

                # Should have logged multiple times due to state changes
                assert mock_debug.call_count >= 2

    @pytest.mark.asyncio
    async def test_do_cancelled_error(self, mock_config, mock_manager):
        """Test do method handling CancelledError"""
        stat = Stat(cfg=mock_config, manager=mock_manager)
        stat.stopper = TaskEvent()

        # Mock sleep to raise CancelledError on first call then set stopper
        call_count = 0
        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.CancelledError()
            # Set stopper to exit loop
            stat.stopper.set()

        with patch('asyncio.sleep', side_effect=mock_sleep):
            with patch('palabra_ai.task.stat.debug') as mock_debug:
                await stat.do()

                # Should log the cancellation attempt
                assert any("tried to cancel" in str(call) for call in mock_debug.call_args_list)

    @pytest.mark.asyncio
    async def test_exit(self, mock_config, mock_manager):
        """Test exit method"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        start_time = time.time()

        with patch('palabra_ai.task.stat.debug') as mock_debug:
            await stat.exit()

            elapsed = time.time() - start_time

            # Should have waited approximately SLEEP_INTERVAL_LONG
            assert elapsed >= SLEEP_INTERVAL_LONG - 0.1

            # Should have logged stats multiple times
            assert mock_debug.call_count >= 2

    @pytest.mark.asyncio
    async def test_exit_cancelled(self, mock_config, mock_manager):
        """Test exit method with cancellation"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        # Mock sleep to raise CancelledError
        call_count = 0
        real_sleep = asyncio.sleep
        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.CancelledError()
            await real_sleep(0.01)

        with patch('asyncio.sleep', side_effect=mock_sleep):
            with patch('palabra_ai.task.stat.debug') as mock_debug:
                # Use short timeout for test
                with patch('palabra_ai.task.stat.SLEEP_INTERVAL_LONG', 0.05):
                    await stat.exit()

                    # Should still complete and log stats
                    assert mock_debug.call_count >= 2

    def test_stat_palabra_tasks(self, mock_config, mock_manager):
        """Test stat_palabra_tasks property"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        # Add mock tasks
        task1 = MagicMock()
        task1.__str__.return_value = "Task1"
        task2 = MagicMock()
        task2.__str__.return_value = "Task2"

        mock_manager.tasks = [task1, task2]

        result = stat.stat_palabra_tasks

        assert "Palabra tasks:" in result
        assert "Task1" in result
        assert "Task2" in result

    def test_stat_asyncio_tasks(self, mock_config, mock_manager):
        """Test stat_asyncio_tasks property"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        # Mock asyncio.all_tasks
        mock_task1 = MagicMock()
        mock_task1.get_name.return_value = "task1"
        mock_task2 = MagicMock()
        mock_task2.get_name.return_value = "task2"

        with patch('asyncio.all_tasks', return_value=[mock_task1, mock_task2]):
            result = stat.stat_asyncio_tasks

            assert "Asyncio tasks:" in result
            assert "task1" in result
            assert "task2" in result

    def test_stat_without_deep_debug(self, mock_config, mock_manager):
        """Test stat property without deep debug"""
        stat = Stat(cfg=mock_config, manager=mock_manager)
        mock_manager.cfg.deep_debug = False

        # Mock asyncio.all_tasks to avoid event loop requirement
        with patch('asyncio.all_tasks', return_value=[]):
            result = stat.stat

            # Should have task info
            assert "Palabra tasks:" in result
            assert "Asyncio tasks:" in result
            # Should not have deep debug info at the start
            assert not result.startswith("deep")

    def test_stat_with_deep_debug(self, mock_config, mock_manager):
        """Test stat property with deep debug"""
        stat = Stat(cfg=mock_config, manager=mock_manager)
        mock_manager.cfg.deep_debug = True

        with patch('palabra_ai.task.stat.diagnose_hanging_tasks', return_value="deep debug info"):
            with patch('asyncio.all_tasks', return_value=[]):
                result = stat.stat

                assert result.startswith("deep debug info")
                assert "Palabra tasks:" in result
                assert "Asyncio tasks:" in result

    def test_banner_property(self, mock_config, mock_manager):
        """Test _banner property"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        # Add mock tasks with states
        task1 = MagicMock()
        task1._state = ["🚀", "🟢"]
        task2 = MagicMock()
        task2._state = ["🎉"]
        task3 = MagicMock()
        task3._state = []  # Empty state

        mock_manager.tasks = [task1, task2, task3]

        result = stat._banner

        assert result == "🟢🎉⭕"

    def test_show_banner(self, mock_config, mock_manager):
        """Test show_banner method"""
        stat = Stat(cfg=mock_config, manager=mock_manager)

        # Mock tasks for banner
        task1 = MagicMock()
        task1._state = ["🚀"]
        mock_manager.tasks = [task1]

        with patch('palabra_ai.task.stat.info') as mock_info:
            stat.show_banner()

            mock_info.assert_called_once_with("🚀")
        """Test run_banner method"""
        stat = Stat(cfg=mock_config, manager=mock_manager)
        stat.sub_tg = MagicMock()

        mock_task = MagicMock()
        stat.sub_tg.create_task.return_value = mock_task

        result = stat.run_banner()

        assert result == mock_task
        stat.sub_tg.create_task.assert_called_once()
        # Check task name
        call_args = stat.sub_tg.create_task.call_args
        assert call_args[1]['name'] == "Stat:info_banner"
