import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from palabra_ai.task.monitor import RtMonitor
from palabra_ai.base.message import Message, TranscriptionMessage
from palabra_ai.config import Config
from palabra_ai.lang import Language
from palabra_ai.task.realtime import RtMsg
from palabra_ai.base.enum import Channel, Direction


class TestTaskRtMonitor:
    @pytest.mark.asyncio
    async def test_silence_property(self):
        cfg = MagicMock(spec=Config)
        rt = MagicMock()
        rt.out_foq.subscribe.return_value = asyncio.Queue()

        monitor = RtMonitor(cfg, rt)

        # Fill with non-transcription messages
        for _ in range(100):
            monitor.msg_history.append(Message(message_type=Message.Type.PIPELINE_TIMINGS))

        assert monitor.silence is True

        # Add transcription message
        monitor.msg_history.append(Message(message_type=Message.Type.PARTIAL_TRANSCRIPTION))
        assert monitor.silence is False

    @pytest.mark.asyncio
    async def test_lifecycle(self):
        cfg = MagicMock(spec=Config)
        rt = MagicMock()
        async def ready_coro(): pass
        rt.ready = ready_coro()
        rt.out_foq.subscribe.return_value = asyncio.Queue()

        monitor = RtMonitor(cfg, rt)

        # Boot
        await monitor.boot()

        # Exit
        await monitor.exit()
        rt.out_foq.unsubscribe.assert_called_once_with(monitor)

    @pytest.mark.asyncio
    async def test_do_processes_messages(self):
        cfg = MagicMock(spec=Config)
        rt = MagicMock()
        q = asyncio.Queue()
        rt.out_foq.subscribe.return_value = q

        monitor = RtMonitor(cfg, rt)

        trans_msg = TranscriptionMessage(
            message_type=Message.Type.PARTIAL_TRANSCRIPTION,
            transcription_id="test123",
            text="Hello world",
            language=Language("en"),
            segments=[]
        )

        rt_msg = RtMsg(Channel.WS, Direction.OUT, trans_msg)
        await q.put(rt_msg)
        await q.put(None)

        await monitor.do()

        assert monitor.msg_counter[Message.Type.PARTIAL_TRANSCRIPTION] >= 1
        assert len(monitor.msg_history) >= 1

    @pytest.mark.asyncio
    async def test_do_timeout(self):
        cfg = MagicMock(spec=Config)
        rt = MagicMock()
        q = asyncio.Queue()
        rt.out_foq.subscribe.return_value = q

        monitor = RtMonitor(cfg, rt)

        async def set_stopper():
            await asyncio.sleep(0.15)
            monitor.stopper.set()

        await asyncio.gather(
            monitor.do(),
            set_stopper(),
            return_exceptions=True
        )
