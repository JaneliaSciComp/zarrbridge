"""Cache location and layout (spec §7.1, §7.2)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

FINGERPRINT = "fingerprint.json"
BUILD_JSON = "build.json"
LOCK = "lock"
INDEX = "index.vzip"
INFO = "info.json"
SOURCE_JSON = "source.json"
BESIDE_DIR = ".zarrbridge"


def central_dir(path: Path, central_root: str | Path) -> Path:
    h = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    return Path(os.path.expanduser(str(central_root))) / h[:2] / h


def beside_dir(path: Path) -> Path:
    path = Path(path)
    return path.parent / BESIDE_DIR / path.name


def cache_dir_for(path: Path, config: Any, mode: str | None = None) -> tuple[Path, str]:
    """(cache_dir, mode). `beside` when the data directory is writable by this process, else `central`."""
    path = Path(path)
    mode = mode or config.cache.mode
    if mode == "beside":
        beside = beside_dir(path)
        if beside.is_dir() or os.access(path.parent, os.W_OK):
            return beside, "beside"
        mode = "central"
    if mode == "central":
        return central_dir(path, config.cache.central_root), "central"
    raise ValueError(f"unknown cache.mode {mode!r}")


def info_key(image_path: str) -> str:
    return f"{image_path}/{INFO}" if image_path else INFO


def cache_bytes(d: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def fsync_tree(d: Path) -> None:
    for root, _dirs, files in os.walk(d):
        for f in files:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
