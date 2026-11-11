"""DatasetResolver (spec §8.1, §8.2): {root}/{path} -> (Dataset, zarr key); LRU of open datasets."""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from zarrbridge.build.queue import AutoBuildQueue
from zarrbridge.dataset import Dataset
from zarrbridge.readers.base import IndexTimeout, ReaderRejected

_log = logging.getLogger("zarrbridge.server")


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


class StillOpening(Exception):
    """The dataset is being opened and was not ready within the wait the caller allowed."""

    def __init__(self, started_at: float) -> None:
        super().__init__("dataset is still opening")
        self.started_at = started_at


class DatasetResolver:
    def __init__(self, config: Any, queue: AutoBuildQueue | None = None) -> None:
        self.config = config
        self.queue = queue
        self.roots = {name: Path(p).resolve() for name, p in config.roots.items()}
        self._lru: "OrderedDict[Path, Dataset]" = OrderedDict()
        self._rejected: dict[Path, tuple[str, float]] = {}
        self._unindexed: dict[Path, tuple[str, float]] = {}
        self._opening: dict[Path, _Opening] = {}
        self._lock = threading.Lock()

    @property
    def open_count(self) -> int:
        return len(self._lru)

    # -- path resolution ---------------------------------------------------------------------------------------
    def root_dir(self, root: str) -> Path:
        try:
            return self.roots[root]
        except KeyError:
            raise NotFound(f"unknown root {root!r}") from None

    def _contained(self, candidate: Path, root: Path) -> bool:
        if self.config.server.follow_symlinks:
            return candidate.is_relative_to(root)
        try:
            return candidate.resolve().is_relative_to(root)
        except OSError:
            return False

    def split(self, root: str, path: str) -> tuple[Path, str]:
        """Longest prefix of `path` that is a regular file under the root; the remainder is the zarr key."""
        base = self.root_dir(root)
        segments = [s for s in path.split("/") if s]
        if any(s in (".", "..") or "\x00" in s for s in segments):
            raise NotFound("bad path")
        cur = base
        for i, seg in enumerate(segments):
            cur = cur / seg
            if not self._contained(cur, base):
                raise NotFound("outside root")
            if cur.is_file():
                return cur, "/".join(segments[i + 1:])
            if not cur.is_dir():
                raise NotFound(path)
        raise NotFound(path)

    def check_trusted(self, abs_path: str) -> Path:
        p = Path(abs_path)
        if not p.is_absolute() or any(s in (".", "..") for s in p.parts):
            raise Forbidden("path must be absolute and canonical")
        prefixes = self.config.server.allow_prefixes
        if prefixes and not any(fnmatch.fnmatch(str(p), g) or str(p).startswith(g.rstrip("*/") + "/") for g in prefixes):
            raise Forbidden("path not under an allowed prefix")
        if not p.is_file():
            raise NotFound(abs_path)
        return p

    # -- datasets ----------------------------------------------------------------------------------------------
    def cached(self, path: Path) -> Dataset | None:
        """The open dataset for `path`, or None. No open, no source check."""
        with self._lock:
            return self._lru.get(Path(path))

    def open_state(self, path: Path) -> tuple[str, dict] | None:
        """What a client should know about a dataset that is not (yet) open: `indexing` while an open is in
        flight, `unindexed` for `limits.open_retry_s` after a timeout, `unservable` for 30 s after a rejection."""
        path = Path(path)
        now = time.time()
        with self._lock:
            op = self._opening.get(path)
            if op is not None:
                return "indexing", {"started_at": op.started_at}
            un = self._unindexed.get(path)
            if un and now - un[1] < self.config.limits.open_retry_s:
                return "unindexed", {"reason": un[0]}
            rej = self._rejected.get(path)
            if rej and now - rej[1] < 30:
                return "unservable", {"reason": rej[0]}
        return None

    def dataset(self, path: Path, wait: float | None = None) -> Dataset:
        """The dataset at `path`, opened on a worker thread once per path however many callers ask. Waits at
        most `wait` seconds (None: until the open ends, which limits.open_timeout_s bounds); raises
        StillOpening when it is not done by then, IndexTimeout / ReaderRejected when the open failed."""
        got = self._start_or_join(Path(path))
        if isinstance(got, Dataset):
            return got
        if not got.done.wait(wait):
            raise StillOpening(got.started_at)
        return got.outcome()

    async def dataset_async(self, path: Path, wait: float | None = None) -> Dataset:
        """`dataset` for coroutines: waits on a future the opener thread resolves, holding no thread."""
        got = self._start_or_join(Path(path))
        if isinstance(got, Dataset):
            return got
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        with self._lock:
            if got.done.is_set():
                fut.set_result(None)
            else:
                got.waiters.append((loop, fut))
        try:
            await asyncio.wait_for(asyncio.shield(fut), wait)
        except asyncio.TimeoutError:
            with self._lock:  # leave: this loop may be gone when the open ends
                if (loop, fut) in got.waiters:
                    got.waiters.remove((loop, fut))
            raise StillOpening(got.started_at) from None
        return got.outcome()

    def _start_or_join(self, path: Path) -> "Dataset | _Opening":
        """The open dataset, or the in-flight open to wait on (started here when there is none). A remembered
        timeout or rejection is raised instead of reopening."""
        with self._lock:
            ds = self._lru.get(path)
            if ds is not None and not ds.source_changed():
                self._lru.move_to_end(path)
                ds.last_used = time.time()
                return ds
            if ds is not None:  # replaced on disk: reopen
                self._lru.pop(path, None)
                ds.close()
            op = self._opening.get(path)
            if op is not None:
                return op
            now = time.time()
            un = self._unindexed.get(path)
            if un and now - un[1] < self.config.limits.open_retry_s:
                raise IndexTimeout(path, self.config.limits.open_timeout_s)
            rej = self._rejected.get(path)
            if rej and now - rej[1] < 30:
                raise ReaderRejected(rej[0])
            op = self._opening[path] = _Opening(now)
        threading.Thread(target=self._open, args=(path, op), name="zarrbridge-open", daemon=True).start()
        return op

    def _open(self, path: Path, op: "_Opening") -> None:
        try:
            ds = Dataset.open(path, self.config)
        except IndexTimeout as e:
            with self._lock:
                self._unindexed[path] = (str(e), time.time())
            op.error = e
        except ReaderRejected as e:
            with self._lock:
                self._rejected[path] = (str(e), time.time())
            op.error = e
        except Exception as e:  # noqa: BLE001 - surface to every waiter rather than hang them
            _log.exception("open %s failed", path)
            op.error = e
        else:
            with self._lock:
                self._unindexed.pop(path, None)
                existing = self._lru.get(path)
                if existing is not None:
                    ds.close()
                    ds = existing
                else:
                    self._lru[path] = ds
                    while len(self._lru) > self.config.limits.max_open_datasets:
                        _, old = self._lru.popitem(last=False)
                        old.close()
            op.result = ds
            if self.config.build.auto:
                ds.ensure_pyramid("auto", self.queue)
        finally:
            with self._lock:
                self._opening.pop(path, None)
                waiters, op.waiters = op.waiters, []
                op.done.set()
            for loop, fut in waiters:
                try:
                    loop.call_soon_threadsafe(_resolve, fut)
                except RuntimeError:  # that waiter's loop is closed; the others still get their answer
                    pass

    def close_idle(self, idle_seconds: float = 600) -> None:
        now = time.time()
        with self._lock:
            for p, ds in list(self._lru.items()):
                if now - ds.last_used > idle_seconds:
                    self._lru.pop(p)
                    ds.close()

    def close_all(self) -> None:
        with self._lock:
            for ds in self._lru.values():
                ds.close()
            self._lru.clear()


def _resolve(fut: "asyncio.Future") -> None:
    if not fut.done():
        fut.set_result(None)


class _Opening:
    """One in-flight Dataset.open: who waits on it and how it ended."""

    def __init__(self, started_at: float) -> None:
        self.started_at = started_at
        self.done = threading.Event()
        self.result: Dataset | None = None
        self.error: BaseException | None = None
        self.waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = []

    def outcome(self) -> Dataset:
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result
