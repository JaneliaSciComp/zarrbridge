"""Run a coroutine to completion from sync code, whether or not a loop is running in this thread."""
from __future__ import annotations

import asyncio
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


def run_sync(coro: Awaitable[T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    return asyncio.run_coroutine_threadsafe(coro, _bg_loop()).result()  # type: ignore[arg-type]
