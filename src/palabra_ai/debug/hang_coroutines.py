import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from palabra_ai.constant import DEBUG_TASK_CHECK_INTERVAL


@dataclass
class TaskInfo:
    """Information about a hanging task"""

    name: str
    coro_name: str
    location: str
    stack_frames: list[tuple[str, int, str, Optional[str]]]
    state: str


def is_user_code(filename: str) -> bool:
    """Check if the file is user code (not stdlib or third-party)"""
    if not filename:
        return False

    # Skip standard library and common async libraries
    skip_patterns = [
        "asyncio/",
        "concurrent/",
        "threading.py",
        "selectors.py",
        "socket.py",
        "ssl.py",
        "site-packages/",
        "dist-packages/",
        "<frozen",
        "<built-in>",
        "<string>",
    ]

    return not any(pattern in filename for pattern in skip_patterns)


def get_meaningful_frames(
    stack: list, max_frames: int = 3
) -> list[tuple[str, int, str, Optional[str]]]:
    """Extract only meaningful frames from stack trace"""
    frames = []

    for frame in reversed(stack):
        filename = frame.f_code.co_filename
        if is_user_code(filename):
            # Get the actual code line
            lineno = frame.f_lineno
            func_name = frame.f_code.co_name

            try:
                with open(filename, encoding="utf-8") as f:
                    lines = f.readlines()
                    code_line = (
                        lines[lineno - 1].strip() if lineno <= len(lines) else None
                    )
            except Exception:
                code_line = None

            frames.append(
                (
                    Path(filename).name,  # Just filename, not full path
                    lineno,
                    func_name,
                    code_line,
                )
            )

            if len(frames) >= max_frames:
                break

    return frames


def diagnose_hanging_tasks() -> str:
    """
    Diagnose hanging asyncio tasks from user code

    Returns:
        Formatted string with diagnosis results
    """
    result = []

    try:
        # Get current event loop
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task(loop)
        all_tasks = asyncio.all_tasks(loop)

        # Collect task information
        hanging_tasks = []

        for task in all_tasks:
            # Skip current task (the one running diagnostics)
            if task is current_task:
                continue

            # Get task info
            coro = task.get_coro()
            task_name = task.get_name()
            coro_name = coro.__name__ if hasattr(coro, "__name__") else str(coro)

            # Get stack
            stack = task.get_stack()
            if not stack:
                continue

            # Get the current location
            current_frame = stack[-1]
            current_file = current_frame.f_code.co_filename

            # Get all frames, not just user code
            frames = []
            for frame in reversed(stack):
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                func_name = frame.f_code.co_name

                try:
                    with open(filename, encoding="utf-8") as f:
                        lines = f.readlines()
                        code_line = (
                            lines[lineno - 1].strip() if lineno <= len(lines) else None
                        )
                except Exception:
                    code_line = None

                frames.append((Path(filename).name, lineno, func_name, code_line))

                if len(frames) >= 3:  # Limit to 3 frames
                    break

            # Current location info
            location = f"{Path(current_file).name}:{current_frame.f_lineno}"

            # Task state
            if task.done():
                state = "DONE"
                if task.cancelled():
                    state = "CANCELLED"
                elif task.exception():
                    state = f"ERROR: {task.exception()}"
            else:
                state = "RUNNING"

            hanging_tasks.append(
                TaskInfo(
                    name=task_name,
                    coro_name=coro_name,
                    location=location,
                    stack_frames=frames,
                    state=state,
                )
            )

        # Build diagnosis string
        if not hanging_tasks:
            result.append("✓ No hanging tasks found")
            return "\n".join(result)

        result.append(f"\n🔍 Found {len(hanging_tasks)} hanging task(s):\n")
        result.append("-" * 60)

        # Group by similar locations
        location_groups = defaultdict(list)
        for task_info in hanging_tasks:
            location_groups[task_info.location].append(task_info)

        # Display grouped tasks
        for location, tasks in location_groups.items():
            result.append(f"\n📍 {location} ({len(tasks)} task(s))")

            for i, task in enumerate(tasks):
                prefix = "  └─" if i == len(tasks) - 1 else "  ├─"
                result.append(f"{prefix} {task.name} [{task.coro_name}] - {task.state}")

                # Show stack trace for first task in group or if different
                if i == 0 or task.stack_frames != tasks[0].stack_frames:
                    for j, (file, line, func, code) in enumerate(task.stack_frames):
                        indent = "     " if i == len(tasks) - 1 else "  │  "
                        arrow = "→" if j == 0 else " "
                        result.append(f"{indent}  {arrow} {file}:{line} in {func}()")
                        if code:
                            result.append(
                                f"{indent}     {code[:50]}{'...' if len(code) > 50 else ''}"
                            )

        result.append("\n" + "-" * 60)

    except RuntimeError:
        result.append("❌ No running event loop. Call from within async context.")

    return "\n".join(result)


async def diagnose_hanging_tasks_async() -> list[TaskInfo]:
    """
    Async version that returns task info as list

    Returns:
        List of TaskInfo objects for further processing
    """
    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task(loop)
    all_tasks = asyncio.all_tasks(loop)

    hanging_tasks = []

    for task in all_tasks:
        if task is current_task:
            continue

        coro = task.get_coro()
        task_name = task.get_name()
        coro_name = coro.__name__ if hasattr(coro, "__name__") else str(coro)

        stack = task.get_stack()
        if not stack:
            continue

        current_frame = stack[-1]
        current_file = current_frame.f_code.co_filename

        # Get all frames
        frames = []
        for frame in reversed(stack):
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            func_name = frame.f_code.co_name

            try:
                with open(filename, encoding="utf-8") as f:
                    lines = f.readlines()
                    code_line = (
                        lines[lineno - 1].strip() if lineno <= len(lines) else None
                    )
            except Exception:
                code_line = None

            frames.append((Path(filename).name, lineno, func_name, code_line))

            if len(frames) >= 3:  # Limit to 3 frames
                break

        location = f"{Path(current_file).name}:{current_frame.f_lineno}"

        if task.done():
            state = "DONE"
            if task.cancelled():
                state = "CANCELLED"
            elif task.exception():
                state = f"ERROR: {task.exception()}"
        else:
            state = "RUNNING"

        hanging_tasks.append(
            TaskInfo(
                name=task_name,
                coro_name=coro_name,
                location=location,
                stack_frames=frames,
                state=state,
            )
        )

    return hanging_tasks


def format_task_info(tasks: list[TaskInfo]) -> str:
    """
    Format list of TaskInfo objects into readable string

    Args:
        tasks: List of TaskInfo objects

    Returns:
        Formatted string representation
    """
    if not tasks:
        return "✓ No hanging tasks found"

    result = []
    result.append(f"\n🔍 Found {len(tasks)} hanging task(s):\n")
    result.append("-" * 60)

    # Group by similar locations
    location_groups = defaultdict(list)
    for task_info in tasks:
        location_groups[task_info.location].append(task_info)

    # Display grouped tasks
    for location, task_list in location_groups.items():
        result.append(f"\n📍 {location} ({len(task_list)} task(s))")

        for i, task in enumerate(task_list):
            prefix = "  └─" if i == len(task_list) - 1 else "  ├─"
            result.append(f"{prefix} {task.name} [{task.coro_name}] - {task.state}")

            # Show stack trace for first task in group or if different
            if i == 0 or task.stack_frames != task_list[0].stack_frames:
                for j, (file, line, func, code) in enumerate(task.stack_frames):
                    indent = "     " if i == len(task_list) - 1 else "  │  "
                    arrow = "→" if j == 0 else " "
                    result.append(f"{indent}  {arrow} {file}:{line} in {func}()")
                    if code:
                        result.append(
                            f"{indent}     {code[:50]}{'...' if len(code) > 50 else ''}"
                        )

    result.append("\n" + "-" * 60)
    return "\n".join(result)


# Example usage and test
if __name__ == "__main__":

    async def hanging_task():
        """Example of hanging task"""
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour

    async def io_waiting_task():
        """Task waiting for I/O"""
        reader, writer = await asyncio.open_connection("example.com", 80)
        data = await reader.read(1024)  # noqa

    async def main():
        # Create some hanging tasks
        task1 = asyncio.create_task(hanging_task(), name="HangingTask-1")
        task2 = asyncio.create_task(hanging_task(), name="HangingTask-2")

        # Wait a bit
        await asyncio.sleep(0.1)

        # Run diagnostics and print result
        print("Running diagnostics...")
        diagnosis = diagnose_hanging_tasks()
        print(diagnosis)

        # Cancel tasks
        task1.cancel()
        task2.cancel()

        try:
            await asyncio.gather(task1, task2)
        except asyncio.CancelledError:
            pass

    # Run example
    asyncio.run(main())


async def monitor_tasks_periodically():
    """Monitor tasks every N seconds"""
    while True:
        try:
            await asyncio.sleep(DEBUG_TASK_CHECK_INTERVAL)

            diagnosis = diagnose_hanging_tasks()
            print(f"\n⏰ Periodic check:{diagnosis}")
        except asyncio.CancelledError:
            print("🛑 Monitor task cancelled, stopping periodic checks.")
            break
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            continue
