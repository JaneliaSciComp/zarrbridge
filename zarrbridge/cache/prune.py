"""`zarrbridge prune` (spec §9.6): delete invalid, orphaned, old or over-budget caches. Never a locked one."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

from zarrbridge.cache.fingerprint import FINGERPRINT, validate
from zarrbridge.cache.layout import BESIDE_DIR, SOURCE_JSON, cache_bytes
from zarrbridge.cache.locks import is_locked
from zarrbridge.readers.base import ReaderRejected
from zarrbridge.readers.registry import open_dataset_reader

_log = logging.getLogger("zarrbridge.prune")


def iter_caches(root: Path) -> Iterator[tuple[Path, Path | None]]:
    """(cache_dir, source_path or None) for every cache under `root` (beside dirs and central layout)."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if FINGERPRINT in filenames or "lock" in filenames or "build.json" in filenames or any(x.isdigit() for x in dirnames):
            if d.parent.name == BESIDE_DIR:
                yield d, d.parent.parent / d.name
                dirnames[:] = []
                continue
            if SOURCE_JSON in filenames:
                try:
                    yield d, Path(json.loads((d / SOURCE_JSON).read_text())["path"])
                except (OSError, ValueError, KeyError):
                    yield d, None
                dirnames[:] = []
                continue


def prune(root: Path, config: Any, *, older_than_s: float | None = None, max_bytes: int | None = None,
          stale_only: bool = False, dry_run: bool = False) -> list[Path]:
    removed: list[Path] = []
    survivors: list[tuple[float, int, Path]] = []
    now = time.time()
    for cache_dir, source in iter_caches(root):
        if is_locked(cache_dir):
            continue
        reason = None
        if source is None or not source.exists():
            reason = "orphaned"
        else:
            try:
                op = open_dataset_reader(source, config)
                try:
                    state, field = validate(cache_dir, op)
                finally:
                    op.close()
            except ReaderRejected:
                state, field = "unservable", None
            if state in ("stale", "outdated", "incompatible", "unservable"):
                reason = f"{state}" + (f" ({field})" if field else "")
            elif state == "missing" and not (cache_dir / "build.json").exists():
                reason = "incomplete"
        try:
            atime = (cache_dir / FINGERPRINT).stat().st_atime
        except OSError:
            atime = now
        if reason is None and older_than_s is not None and now - atime > older_than_s:
            reason = f"older than {older_than_s:.0f}s"
        if reason:
            _log.info("prune %s: %s", cache_dir, reason)
            removed.append(cache_dir)
            if not dry_run:
                shutil.rmtree(cache_dir, ignore_errors=True)
        elif not stale_only:
            survivors.append((atime, cache_bytes(cache_dir), cache_dir))
    if max_bytes is not None and not stale_only:
        total = sum(b for _, b, _ in survivors)
        for atime, b, d in sorted(survivors):
            if total <= max_bytes:
                break
            _log.info("prune %s: over --max-bytes (LRU)", d)
            removed.append(d)
            total -= b
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)
    return removed
