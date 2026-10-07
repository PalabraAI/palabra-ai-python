import asyncio
import copy
import json

import pytest
import websockets

from palabra_ai import NotReadyError, Palabra, Region, SessionError, StreamEnd, TaskInfo, TaskStatus
from palabra_ai.events import parse_event
from test_sts_session import FakeServer, _fake_region


@pytest.mark.parametrize(
    "command,event,state",
    [
        ("set_task", "started", "running"),
        ("set_task", "updated", "running"),
        ("set_task", "unchanged", "running"),
        ("set_task", "resumed", "running"),
        ("pause_task", "paused", "paused"),
        ("interrupt_task", "interrupted", "running"),
        ("tts_task", "queued", "paused"),
    ],
)
def test_parse_status(command, event, state):
    parsed = parse_event(
        {
            "message_type": "task_status",
            "data": {
                "command": command,
                "event": event,
                "task_status": state,
                **({"transcription_id": "tts1"} if command == "tts_task" else {}),
            },
        }
    )
    assert parsed == TaskStatus(
        "task_status", command, event, state, "tts1" if command == "tts_task" else None
    )


async def test_commands_are_observable_and_startup_needs_no_polling(monkeypatch):
    async with FakeServer() as server:
        _fake_region(monkeypatch, server)
        async with Palabra(api_key="test").translation("en", "es") as session:
            assert server.commands == ["set_task"]
            assert await session.receive(timeout=1) == TaskStatus(
                "task_status", "set_task", "started", "running"
            )
            await session.set_task(session.task)
            updated_task = copy.deepcopy(session.task)
            updated_task["pipeline"]["allowed_message_types"] = ["output_audio_data"]
            await session.set_task(updated_task)
            await session.pause()
            await session.resume()
            await session.interrupt(pause=True)
            await session.speak("Hola", "es")
            await session.end()
            events = [event async for event in session]
        assert [(e.command, e.event, e.status) for e in events if isinstance(e, TaskStatus)] == [
            ("set_task", "unchanged", "running"),
            ("set_task", "updated", "running"),
            ("pause_task", "paused", "paused"),
            ("set_task", "resumed", "running"),
            ("interrupt_task", "interrupted", "paused"),
            ("tts_task", "queued", "paused"),
        ]
        assert events[-2].transcription_id == "tts1"
        assert isinstance(events[-1], StreamEnd)


async def test_get_task_is_available_when_explicitly_requested(monkeypatch):
    async with FakeServer() as server:
        _fake_region(monkeypatch, server)
        async with Palabra(api_key="test").translation("en", "es") as session:
            assert isinstance(await session.receive(timeout=1), TaskStatus)
            await session.get_task()
            task_info = await session.receive(timeout=1)
            assert isinstance(task_info, TaskInfo)
            assert task_info.status == "running"
            assert server.commands == ["set_task", "get_task"]
            await session.end()
            assert isinstance(await session.receive(timeout=1), StreamEnd)


async def test_missing_status_times_out_without_polling(monkeypatch):
    monkeypatch.setattr("palabra_ai.client.READY_TIMEOUT", 0.1)
    commands = []
    closed = asyncio.Event()

    async def handler(ws):
        async for raw in ws:
            commands.append(json.loads(raw)["message_type"])
        closed.set()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr("palabra_ai.client.REGIONS", {"eu": Region(translation=f"ws://127.0.0.1:{port}")})
        with pytest.raises(NotReadyError, match="did not confirm"):
            async with Palabra(api_key="test").translation("en", "es"):
                pytest.fail("Missing task_status must not confirm set_task")
        await asyncio.wait_for(closed.wait(), 1)
        assert commands == ["set_task"]


async def test_slow_start_waits_for_status_without_polling(monkeypatch):
    monkeypatch.setattr("palabra_ai.client.READY_TIMEOUT", 5)
    commands = []

    async def handler(ws):
        async for raw in ws:
            command = json.loads(raw)["message_type"]
            commands.append(command)
            if command == "set_task":
                await asyncio.sleep(2.2)  # longer than the old get_task fallback delay
                await ws.send(
                    json.dumps(
                        {
                            "message_type": "task_status",
                            "data": {
                                "command": "set_task",
                                "event": "started",
                                "task_status": "running",
                            },
                        }
                    )
                )
            elif command == "end_task":
                await ws.close()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr("palabra_ai.client.REGIONS", {"eu": Region(translation=f"ws://127.0.0.1:{port}")})
        async with Palabra(api_key="test").translation("en", "es") as session:
            await session.end()
            async for _ in session:
                pass
        assert commands == ["set_task", "end_task"]


@pytest.mark.parametrize(
    "behavior", ["close", "wrong_command", "wrong_event", "paused", "stale_current_task"]
)
async def test_setup_does_not_accept_unrelated_events(monkeypatch, behavior):
    monkeypatch.setattr("palabra_ai.client.READY_TIMEOUT", 0.15)

    async def handler(ws):
        await ws.recv()
        if behavior == "close":
            await ws.close()
            return
        if behavior != "stale_current_task":
            msg = {
                "message_type": "task_status",
                "data": {
                    "command": "interrupt_task" if behavior == "wrong_command" else "set_task",
                    "event": "queued" if behavior == "wrong_event" else "started",
                    "task_status": "paused" if behavior == "paused" else "running",
                },
            }
        else:
            msg = {"message_type": "current_task", "data": {"task_status": "running"}}
        await ws.send(json.dumps(msg))
        await ws.wait_closed()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr("palabra_ai.client.REGIONS", {"eu": Region(translation=f"ws://127.0.0.1:{port}")})
        with pytest.raises(SessionError if behavior == "close" else NotReadyError):
            async with Palabra(api_key="test").translation("en", "es"):
                pytest.fail("Unrelated messages must not confirm our set_task")


@pytest.mark.parametrize("event", ["started", "updated", "unchanged", "resumed"])
async def test_start_ack_waits_without_consuming_stream_events(monkeypatch, event):
    applied = asyncio.Event()

    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"message_type": "warning", "data": {"code": "TEST", "message": "test"}}))
        await applied.wait()
        await ws.send(
            json.dumps(
                {
                    "message_type": "task_status",
                    "data": {
                        "command": "set_task",
                        "event": event,
                        "task_status": "running",
                    },
                }
            )
        )
        await ws.wait_closed()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr("palabra_ai.client.REGIONS", {"eu": Region(translation=f"ws://127.0.0.1:{port}")})
        session = Palabra(api_key="test").translation("en", "es")
        enter = asyncio.create_task(session.__aenter__())
        try:
            await asyncio.sleep(0.05)
            assert not enter.done()
            applied.set()
            await asyncio.wait_for(enter, 1)
            assert (await session.receive(timeout=1)).type == "warning"
            assert await session.receive(timeout=1) == TaskStatus("task_status", "set_task", event, "running")
        finally:
            enter.cancel()
            await asyncio.gather(enter, return_exceptions=True)
            await session.close()
