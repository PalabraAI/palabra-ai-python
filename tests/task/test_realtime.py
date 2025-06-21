import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from palabra_ai.task.realtime import Realtime, RtMsg
from palabra_ai.config import Config
from palabra_ai.base.enum import Channel, Direction


class TestTaskRealtime:
    @pytest.mark.asyncio
    async def test_reroute(self):
        cfg = MagicMock(spec=Config)
        creds = MagicMock()
        rt = Realtime(cfg, creds)

        from_q = asyncio.Queue()
        to_foq = MagicMock()
        to_foq.publish = MagicMock()

        # Add messages
        await from_q.put("test_msg")
        await from_q.put(None)

        await rt._reroute(Channel.WS, Direction.IN, from_q, [to_foq])

        assert to_foq.publish.call_count >= 1
        call_args = to_foq.publish.call_args_list[0][0][0]
        assert isinstance(call_args, RtMsg)
        assert call_args.ch == Channel.WS
        assert call_args.dir == Direction.IN
        assert call_args.msg == "test_msg"

    @pytest.mark.asyncio
    async def test_exit_publishes_none(self):
        cfg = MagicMock(spec=Config)
        creds = MagicMock()
        rt = Realtime(cfg, creds)

        rt.c = MagicMock()
        rt.c.close = AsyncMock()

        # Mock publish methods
        rt.in_foq.publish = MagicMock()
        rt.out_foq.publish = MagicMock()
        rt.ws_in_foq.publish = MagicMock()
        rt.ws_out_foq.publish = MagicMock()
        rt.webrtc_out_foq.publish = MagicMock()

        await rt.exit()

        # Should publish None to all queues
        rt.in_foq.publish.assert_called_with(None)
        rt.out_foq.publish.assert_called_with(None)
        rt.ws_in_foq.publish.assert_called_with(None)
        rt.ws_out_foq.publish.assert_called_with(None)
        rt.webrtc_out_foq.publish.assert_called_with(None)

    @pytest.mark.asyncio
    async def test_do_with_stopper(self):
        cfg = MagicMock(spec=Config)
        creds = MagicMock()
        rt = Realtime(cfg, creds)

        rt.stopper.set()
        await rt.do()  # Should exit immediately

    @pytest.mark.asyncio
    async def test_reroute_timeout(self):
        cfg = MagicMock(spec=Config)
        creds = MagicMock()
        rt = Realtime(cfg, creds)

        from_q = asyncio.Queue()
        to_foq = MagicMock()

        async def stop_soon():
            await asyncio.sleep(0.1)
            rt.stopper.set()

        await asyncio.gather(
            rt._reroute("ws", "in", from_q, [to_foq]),
            stop_soon(),
            return_exceptions=True
        )
