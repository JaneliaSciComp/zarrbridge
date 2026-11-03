"""Bounded LRU of open fds keyed by (path, size, mtime_ns); refcounted so an evicted fd whose
number was recycled can never serve another file's bytes (port of mrc-ng-server fdcache.py).
Generic: `parse` turns an fd into whatever per-file state the reader needs (an MrcHeader, a TiffFile...)."""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

_logger = logging.getLogger("zarrbridge.fdcache")


class _Entry:
    __slots__ = ("fd", "state", "refs", "evicted")

    def __init__(self, fd: int, state: Any):
        self.fd, self.state, self.refs, self.evicted = fd, state, 0, False


class Handle:
    __slots__ = ("fd", "state")

    def __init__(self, fd: int, state: Any):
        self.fd, self.state = fd, state

    def __iter__(self):
        return iter((self.fd, self.state))


class FdCache:
    def __init__(self, parse: Callable[[int, os.stat_result, Path], Any] | None = None, max_size: int = 256):
        self._parse = parse or (lambda fd, st, path: None)
        self._max_size = max_size
        self._entries: "OrderedDict[tuple, _Entry]" = OrderedDict()
        self._lock = threading.Lock()
        self._evictions = 0

    @staticmethod
    def _key_for(path: Path) -> tuple:
        st = os.stat(path)
        return (str(path), st.st_size, st.st_mtime_ns)

    @contextmanager
    def open(self, path: Path) -> Iterator[Handle]:
        entry = self._acquire(Path(path))
        try:
            yield Handle(entry.fd, entry.state)
        finally:
            self._release(entry)

    def _acquire(self, path: Path) -> _Entry:
        key = self._key_for(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                entry.refs += 1
                return entry
        fd = os.open(str(path), os.O_RDONLY)
        try:
            state = self._parse(fd, os.fstat(fd), path)
        except BaseException:
            os.close(fd)
            raise
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                os.close(fd)
                self._entries.move_to_end(key)
                existing.refs += 1
                return existing
            entry = _Entry(fd, state)
            entry.refs = 1
            self._entries[key] = entry
            while len(self._entries) > self._max_size:
                _, old = self._entries.popitem(last=False)
                old.evicted = True
                if old.refs == 0:
                    os.close(old.fd)
                self._evictions += 1
                if self._evictions % self._max_size == 0:
                    _logger.warning("fd cache evicted %d entries (size %d)", self._evictions, self._max_size)
            return entry

    def _release(self, entry: _Entry) -> None:
        with self._lock:
            entry.refs -= 1
            if entry.refs == 0 and entry.evicted:
                os.close(entry.fd)

    def close_all(self) -> None:
        with self._lock:
            for e in self._entries.values():
                e.evicted = True
                if e.refs == 0:
                    os.close(e.fd)
            self._entries.clear()
