"""Build lock: `lock` file holding host:pid:started_at under flock (spec §9.5)."""
from __future__ import annotations

import fcntl
import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from zarrbridge.cache.layout import LOCK, make_cache_dir

_log = logging.getLogger("zarrbridge.build")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_info(cache_dir: Path) -> tuple[str, int, str] | None:
    try:
        host, pid, started = (Path(cache_dir) / LOCK).read_text().strip().split(":", 2)
        return host, int(pid), started
    except (OSError, ValueError):
        return None


def is_locked(cache_dir: Path) -> bool:
    p = Path(cache_dir) / LOCK
    if not p.exists():
        return False
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


class BuildLock:
    def __init__(self, fd: int, path: Path) -> None:
        self.fd, self.path = fd, path

    @classmethod
    def acquire(cls, cache_dir: Path, stale_seconds: float = 6 * 3600) -> "BuildLock | None":
        cache_dir = Path(cache_dir)
        make_cache_dir(cache_dir)
        path = cache_dir / LOCK
        for attempt in range(2):
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                if attempt == 0 and cls._break_if_stale(path, stale_seconds):
                    continue
                return None
            os.ftruncate(fd, 0)
            started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            os.write(fd, f"{socket.gethostname()}:{os.getpid()}:{started}".encode())
            os.fsync(fd)
            return cls(fd, path)
        return None

    @staticmethod
    def _break_if_stale(path: Path, stale_seconds: float) -> bool:
        info = lock_info(path.parent)
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        if info is None:
            return age > stale_seconds
        host, pid, _ = info
        if host == socket.gethostname() and not _pid_alive(pid) and age > stale_seconds:
            _log.warning("breaking stale build lock %s (pid %d dead, age %.0fs)", path, pid, age)
            try:
                path.unlink()
            except OSError:
                return False
            return True
        return False

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "BuildLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
