import asyncio
import pytest
from unittest.mock import MagicMock, patch
from palabra_ai.task.base import Task, TaskEvent
from palabra_ai.config import Config


class TestTaskEvent:
    """Test TaskEvent class"""

    def test_init(self):
        """Test TaskEvent initialization"""
        event = TaskEvent()
        assert isinstance(event, asyncio.Event)
        assert event._owner == ""
        assert not event.is_set()

    def test_set_owner(self):
        """Test set_owner method"""
        event = TaskEvent()
        event.set_owner("test.event")
        assert event._owner == "test.event"

    def test_log(self):
        """Test log method"""
        event = TaskEvent()
        event.set_owner("test.event")

        with patch('palabra_ai.task.base.debug') as mock_debug:
            event.log()
            mock_debug.assert_called_once_with("[-] test.event")

        event.set()
        with patch('palabra_ai.task.base.debug') as mock_debug:
            event.log()
            mock_debug.assert_called_once_with("[+] test.event")

    def test_pos_operator(self):
        """Test __pos__ operator (unary +)"""
        event = TaskEvent()
        event.set_owner("test.event")

        with patch('palabra_ai.task.base.debug') as mock_debug:
            result = +event
            assert result == event
            assert event.is_set()
            mock_debug.assert_called_once_with("[+] test.event")

    def test_neg_operator(self):
        """Test __neg__ operator (unary -)"""
        event = TaskEvent()
        event.set_owner("test.event")
        event.set()

        with patch('palabra_ai.task.base.debug') as mock_debug:
            result = -event
            assert result == event
            assert not event.is_set()
            mock_debug.assert_called_once_with("[-] test.event")

    def test_bool(self):
        """Test __bool__ operator"""
        event = TaskEvent()
        assert not bool(event)

        event.set()
        assert bool(event)

    @pytest.mark.asyncio
    async def test_await_not_set(self):
        """Test awaiting event when not set"""
        event = TaskEvent()

        async def set_event():
            await asyncio.sleep(0.01)
            event.set()

        asyncio.create_task(set_event())
        await event
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_await_already_set(self):
        """Test awaiting event when already set"""
        event = TaskEvent()
        event.set()

        # Should return immediately
        await event
        assert event.is_set()

    def test_repr(self):
        """Test __repr__ method"""
        event = TaskEvent()
        assert repr(event) == "TaskEvent(False)"

        event.set()
        assert repr(event) == "TaskEvent(True)"


class ConcreteTask(Task):
    """Concrete implementation of Task for testing"""

    async def boot(self):
        pass

    async def do(self):
        pass

    async def exit(self):
        pass


class TestTask:
    """Test Task abstract base class"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock(spec=Config)
        return config

    def test_init(self, mock_config):
        """Test Task initialization"""
        task = ConcreteTask(cfg=mock_config)
        assert task.cfg == mock_config
        assert isinstance(task.ready, TaskEvent)
        assert isinstance(task.eof, TaskEvent)
        assert isinstance(task.stopper, TaskEvent)
        assert task._state == []
        assert task._name is None
        assert task._task is None

    def test_call(self, mock_config):
        """Test __call__ method"""
        task = ConcreteTask(cfg=mock_config)
        mock_tg = MagicMock(spec=asyncio.TaskGroup)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_tg.create_task.return_value = mock_task

        result = task(mock_tg)

        assert result == task
        assert task.root_tg == mock_tg
        assert task._task == mock_task
        assert task.ready._owner == "[T]ConcreteTask.ready"
        assert task.eof._owner == "[T]ConcreteTask.eof"
        assert task.stopper._owner == "[T]ConcreteTask.stopper"
        mock_tg.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success(self, mock_config):
        """Test successful run"""
        task = ConcreteTask(cfg=mock_config)
        task.root_tg = MagicMock()

        with patch('palabra_ai.task.base.debug') as mock_debug:
            result = await task.run()

            # Check state progression
            assert "🚀" in task._state
            assert "🌀" in task._state
            assert "🟢" in task._state
            assert "💫" in task._state
            assert "🎉" in task._state
            assert "👋" in task._state
            assert "🟠" in task._state

            # Check ready and stopper are set
            assert task.ready.is_set()
            assert task.stopper.is_set()

            # Check debug messages
            assert any("starting..." in str(call) for call in mock_debug.call_args_list)
            assert any("ready, doing..." in str(call) for call in mock_debug.call_args_list)
            assert any("done, exiting..." in str(call) for call in mock_debug.call_args_list)

    @pytest.mark.asyncio
    async def test_run_cancelled(self, mock_config):
        """Test run when cancelled"""
        task = ConcreteTask(cfg=mock_config)
        task.root_tg = MagicMock()

        # Make do() raise CancelledError
        async def cancelled_do():
            raise asyncio.CancelledError()

        task.do = cancelled_do

        with patch('palabra_ai.task.base.debug') as mock_debug:
            with pytest.raises(asyncio.CancelledError):
                await task.run()

            # Check state includes cancelled
            assert "🚫" in task._state
            assert task.stopper.is_set()

            # Check debug message
            assert any("cancelled, exiting..." in str(call) for call in mock_debug.call_args_list)

    @pytest.mark.asyncio
    async def test_run_exception(self, mock_config):
        """Test run with exception"""
        task = ConcreteTask(cfg=mock_config)
        task.root_tg = MagicMock()
        task.sub_tg = MagicMock()

        # Make do() raise exception
        async def failing_do():
            raise ValueError("Test error")

        task.do = failing_do

        with patch('palabra_ai.task.base.error') as mock_error:
            with pytest.raises(ValueError, match="Test error"):
                await task.run()

            # Check state includes error
            assert "💥" in task._state
            assert task.stopper.is_set()

            # Check error logging (called twice: once for error msg, once for traceback)
            assert mock_error.call_count == 2
            assert "failed with error" in str(mock_error.call_args_list[0])
            assert "full traceback" in str(mock_error.call_args_list[1])

            # Check abort was called
            task.sub_tg._abort.assert_called_once()
            task.root_tg._abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_boot_delegation(self, mock_config):
        """Test _boot delegates to boot"""
        task = ConcreteTask(cfg=mock_config)
        task.boot = MagicMock(return_value=asyncio.Future())
        task.boot.return_value.set_result("boot_result")

        result = await task._boot()

        assert result == "boot_result"
        task.boot.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_success(self, mock_config):
        """Test _exit with successful exit"""
        task = ConcreteTask(cfg=mock_config)

        async def mock_exit():
            return "exit_result"

        task.exit = mock_exit

        with patch('palabra_ai.task.base.debug') as mock_debug:
            result = await task._exit()

            assert result == "exit_result"
            assert any("waiting for exit..." in str(call) for call in mock_debug.call_args_list)

    @pytest.mark.asyncio
    async def test_exit_timeout(self, mock_config):
        """Test _exit with timeout"""
        task = ConcreteTask(cfg=mock_config)

        async def slow_exit():
            await asyncio.sleep(10)

        task.exit = slow_exit
        task._task = MagicMock()

        with patch('palabra_ai.task.base.error') as mock_error:
            with patch('palabra_ai.task.base.warning') as mock_warning:
                with patch('palabra_ai.task.base.SHUTDOWN_TIMEOUT', 0.01):
                    with patch.object(task, 'cancel_all_subtasks') as mock_cancel:
                        result = await task._exit()

                        assert result is None
                        mock_error.assert_called_once()
                        assert "timed out" in str(mock_error.call_args)
                        mock_cancel.assert_called_once()
                        mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_all_subtasks(self, mock_config):
        """Test cancel_all_subtasks method"""
        task = ConcreteTask(cfg=mock_config)
        task._name = "TestTask"
        task._task = MagicMock()

        # Create mock subtasks
        subtask1 = MagicMock()
        subtask1.get_name.return_value = "[T]TestTask.subtask1"
        subtask1.done.return_value = False

        subtask2 = MagicMock()
        subtask2.get_name.return_value = "[T]TestTask.subtask2"
        subtask2.done.return_value = True  # Already done

        other_task = MagicMock()
        other_task.get_name.return_value = "[T]OtherTask"

        with patch('asyncio.all_tasks', return_value=[task._task, subtask1, subtask2, other_task]):
            with patch('asyncio.wait', return_value=([], [])) as mock_wait:
                with patch('palabra_ai.task.base.debug') as mock_debug:
                    await task.cancel_all_subtasks()

                    # Only subtask1 should be cancelled
                    subtask1.cancel.assert_called_once()
                    subtask2.cancel.assert_not_called()
                    other_task.cancel.assert_not_called()

                    # Check debug message
                    assert any("Cancelling subtask" in str(call) for call in mock_debug.call_args_list)

    def test_name_property(self, mock_config):
        """Test name property"""
        task = ConcreteTask(cfg=mock_config)
        assert task.name == "[T]ConcreteTask"

        task._name = "CustomName"
        assert task.name == "[T]CustomName"

    def test_name_setter(self, mock_config):
        """Test name setter"""
        task = ConcreteTask(cfg=mock_config)
        task.name = "NewName"
        assert task._name == "NewName"
        assert task.name == "[T]NewName"

    def test_task_property_not_set(self, mock_config):
        """Test task property when not set"""
        task = ConcreteTask(cfg=mock_config)

        with pytest.raises(RuntimeError, match="task not set"):
            _ = task.task

    def test_task_property_set(self, mock_config):
        """Test task property when set"""
        task = ConcreteTask(cfg=mock_config)
        mock_task = MagicMock()
        task._task = mock_task

        assert task.task == mock_task

    def test_str_deep_debug(self, mock_config):
        """Test __str__ with DEEP_DEBUG=True"""
        task = ConcreteTask(cfg=mock_config)
        task._name = "Test"
        task.ready.set()
        task.stopper.set()
        task._state = ["🚀", "🟢"]

        with patch('palabra_ai.config.DEEP_DEBUG', True):
            result = str(task)
            assert "[T]Test" in result
            assert "ready=" in result
            assert "stopper=" in result
            assert "eof=" in result
            assert "states=🚀🟢" in result

    def test_str_normal(self, mock_config):
        """Test __str__ with DEEP_DEBUG=False"""
        task = ConcreteTask(cfg=mock_config)
        task._name = "Test"
        task.ready.set()
        task._state = ["🚀", "🟢"]

        with patch('palabra_ai.config.DEEP_DEBUG', False):
            with patch('palabra_ai.task.base.Emoji.bool', side_effect=lambda x: "✅" if x else "❌"):
                result = str(task)
                assert "[T]Test" in result
                assert "🎬" in result  # ready
                assert "🪦" in result  # stopper
                assert "🏁" in result  # eof
                assert "🚀🟢" in result  # states
