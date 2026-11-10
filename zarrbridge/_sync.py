"""Run a coroutine to completion from sync code, whether or not a loop is running in this thread."""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Awaitable, TypeVar

T = TypeVar("T")
_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def _bg_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="zarrbridge-sync", daemon=True).start()
        return _loop


def run_sync_with_timeout(coro: Awaitable[T], timeout: float | None) -> T:
    """`coro` on the background loop, the caller waiting at most `timeout` seconds (None: forever).
    On timeout the task is cancelled before TimeoutError is raised: awaits inside it stop being
    scheduled; work already running on an executor thread finishes on its own."""
    fut = asyncio.run_coroutine_threadsafe(coro, _bg_loop())  # type: ignore[arg-type]
    try:
        return fut.result(timeout)
    except concurrent.futures.TimeoutError:
        fut.cancel()
        raise TimeoutError(f"not done within {timeout} s") from None


def run_sync(coro: Awaitable[T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    return asyncio.run_coroutine_threadsafe(coro, _bg_loop()).result()  # type: ignore[arg-type]
