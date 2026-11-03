"""DatasetResolver (spec §8.1, §8.2): {root}/{path} -> (Dataset, zarr key); LRU of open datasets."""
from __future__ import annotations

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
from zarrbridge.readers.base import ReaderRejected

_log = logging.getLogger("zarrbridge.server")


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


class DatasetResolver:
    def __init__(self, config: Any, queue: AutoBuildQueue | None = None) -> None:
        self.config = config
        self.queue = queue
        self.roots = {name: Path(p).resolve() for name, p in config.roots.items()}
        self._lru: "OrderedDict[Path, Dataset]" = OrderedDict()
        self._rejected: dict[Path, tuple[str, float]] = {}
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
    def dataset(self, path: Path) -> Dataset:
        path = Path(path)
        with self._lock:
            ds = self._lru.get(path)
            if ds is not None and not ds.source_changed():
                self._lru.move_to_end(path)
                ds.last_used = time.time()
                return ds
            if ds is not None:  # replaced on disk: reopen
                self._lru.pop(path, None)
                ds.close()
            rej = self._rejected.get(path)
            if rej and time.time() - rej[1] < 30:
                raise ReaderRejected(rej[0])
        try:
            ds = Dataset.open(path, self.config)
        except ReaderRejected as e:
            with self._lock:
                self._rejected[path] = (str(e), time.time())
            raise
        with self._lock:
            existing = self._lru.get(path)
            if existing is not None:
                ds.close()
                return existing
            self._lru[path] = ds
            while len(self._lru) > self.config.limits.max_open_datasets:
                _, old = self._lru.popitem(last=False)
                old.close()
        if self.config.build.auto:
            ds.ensure_pyramid("auto", self.queue)
        return ds

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
