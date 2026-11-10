"""AutoBuildQueue (spec §8.5): spawn `zarrbridge build` subprocesses, bounded by slots, deduped by cache dir."""
from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger("zarrbridge.build")


class AutoBuildQueue:
    def __init__(self, config: Any, popen=subprocess.Popen) -> None:
        self.config = config
        self._popen = popen
        self._q: "queue.Queue[tuple[Path, Path, str] | None]" = queue.Queue()
        self._pending: set[Path] = set()
        self._running: dict[Path, Any] = {}
        self._done: dict[Path, int] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max(1, int(config.build.slots)))
        self._thread = threading.Thread(target=self._run, name="zarrbridge-build-queue", daemon=True)
        self._thread.start()

    def enqueue(self, path: Path, cache_dir: Path, cache_mode: str) -> bool:
        with self._lock:
            if cache_dir in self._pending or cache_dir in self._running:
                return False
            self._pending.add(cache_dir)
            self._done.pop(cache_dir, None)
        self._q.put((Path(path), Path(cache_dir), cache_mode))
        return True

    def pending(self, cache_dir: Path) -> bool:
        with self._lock:
            return cache_dir in self._pending or cache_dir in self._running

    def failure(self, cache_dir: Path) -> int | None:
        """Exit code of the last auto-build of this cache dir if it failed, else None.
        0 built, 3 skipped (locked), 4 skipped (already valid) are not failures."""
        with self._lock:
            rc = self._done.get(cache_dir)
        return rc if rc is not None and rc not in (0, 3, 4) else None

    def snapshot(self) -> dict:
        with self._lock:
            return {"pending": len(self._pending), "running": len(self._running),
                    "finished": dict((str(k), v) for k, v in list(self._done.items())[-20:])}

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, cache_dir, mode = item
            self._slots.acquire()
            argv = [sys.executable, "-m", "zarrbridge", "--cache-mode", mode, "build", str(path)]
            if self.config.build.nice:
                argv = ["nice", "-n", str(self.config.build.nice)] + argv
            try:
                proc = self._popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            except OSError as e:
                _log.error("auto-build spawn failed for %s: %s", path, e)
                with self._lock:
                    self._pending.discard(cache_dir)
                    self._done[cache_dir] = 1
                self._slots.release()
                continue
            with self._lock:
                self._pending.discard(cache_dir)
                self._running[cache_dir] = proc
            threading.Thread(target=self._wait, args=(proc, path, cache_dir), daemon=True).start()

    def _wait(self, proc, path: Path, cache_dir: Path) -> None:
        try:
            _, err = proc.communicate()
            rc = proc.returncode
            if rc not in (0, 3, 4):
                _log.warning("auto-build of %s exited %s: %s", path, rc, (err or "").strip()[-500:])
            else:
                _log.info("auto-build of %s exited %s", path, rc)
        finally:
            with self._lock:
                self._running.pop(cache_dir, None)
                self._done[cache_dir] = proc.returncode
            self._slots.release()

    def shutdown(self) -> None:
        self._q.put(None)
