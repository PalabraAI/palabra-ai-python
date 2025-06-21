import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from palabra_ai.base.task import Task
from palabra_ai.base.task_event import TaskEvent
from tests.conftest import BaseTaskTest


class ConcreteTask(Task):
    """Concrete implementation for testing."""

    async def boot(self):
        await asyncio.sleep(0.01)

    async def do(self):
        await asyncio.sleep(0.01)

    async def exit(self):
        return "exit_result"


class TestTask(BaseTaskTest):
    @pytest.mark.asyncio
    async def test_cancel_all_subtasks(self):
        task = ConcreteTask()

        # Mock subtasks
        mock_task1 = MagicMock()
        mock_task1.get_name.return_value = "[T]ConcreteTask_subtask1"
        mock_task1.done.return_value = False
        mock_task1.cancel = MagicMock()

        mock_task2 = MagicMock()
        mock_task2.get_name.return_value = "[T]ConcreteTask_subtask2"
        mock_task2.done.return_value = True

        with patch('asyncio.all_tasks', return_value={mock_task1, mock_task2}):
            with patch('asyncio.wait', return_value=(set(), {mock_task1})):
                await task.cancel_all_subtasks()

        # Only non-done task should be cancelled
        mock_task1.cancel.assert_called()
        assert mock_task1.cancel.call_count == 2

    @pytest.mark.asyncio
    async def test_exit_timeout(self):
        class SlowExitTask(Task):
            async def boot(self): pass
            async def do(self): pass
            async def exit(self):
                await asyncio.sleep(10)

        task = SlowExitTask()
        task.cancel_all_subtasks = AsyncMock()

        with patch('palabra_ai.constant.SHUTDOWN_TIMEOUT', 0.1):
            await task._exit()

        task.cancel_all_subtasks.assert_called_once()

    def test_name_setter(self):
        task = ConcreteTask()
        task.name = "CustomName"
        assert task.name == "[T]CustomName"

    def test_task_not_set_error(self):
        task = ConcreteTask()
        with pytest.raises(RuntimeError, match="task not set"):
            _ = task.task

    def test_str_deep_debug(self):
        task = ConcreteTask()
        task._state = ["🚀"]

        with patch('palabra_ai.base.task.DEEP_DEBUG', True):
            str_repr = str(task)
            assert "ready=" in str_repr
            assert "stopper=" in str_repr
            assert "eof=" in str_repr
            assert "states=" in str_repr

    def test_str_normal_mode(self):
        task = ConcreteTask()
        task._state = ["🚀", "🌀"]

        with patch('palabra_ai.base.task.DEEP_DEBUG', False):
            str_repr = str(task)
            assert "🎬" in str_repr
            assert "🪦" in str_repr
            assert "🏁" in str_repr
            assert "🚀🌀" in str_repr


class TestTaskEvent:
    def test_bool_operators(self):
        event = TaskEvent()
        assert not event
        event.set()
        assert event

    @pytest.mark.asyncio
    async def test_await_functionality(self):
        event = TaskEvent()

        # When already set
        event.set()
        await event  # Should not block

        # Test repr
        assert "TaskEvent(True)" in repr(event)

    def test_pos_neg_operators(self):
        event = TaskEvent()
        event.set_owner("test.event")

        # Test __pos__ (set)
        +event
        assert event.is_set()

        # Test __neg__ (clear)
        -event
        assert not event.is_set()
