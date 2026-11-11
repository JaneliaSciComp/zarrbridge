"""Cache location and layout (spec §7.1, §7.2)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

FINGERPRINT = "fingerprint.json"
BUILD_JSON = "build.json"
LOCK = "lock"
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


def make_cache_dir(cache_dir: Path) -> None:
    """mkdir -p. A `.zarrbridge` directory this creates gets setgid and group write (02775) so every cache
    under it, whoever builds it, keeps the data directory's group and stays rebuildable by the lab."""
    cache_dir = Path(cache_dir)
    root = cache_dir.parent
    if root.name == BESIDE_DIR and not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(exist_ok=True)
        try:
            os.chmod(root, 0o2775)
        except OSError:  # not ours after all (lost a race), or a filesystem without setgid
            pass
    cache_dir.mkdir(parents=True, exist_ok=True)


def cache_writable(cache_dir: Path) -> bool:
    """Whether this process can write the cache: the directory, or the nearest existing parent when it does not
    exist yet. False for a `beside` cache another user built with a strict umask."""
    d = Path(cache_dir)
    while not d.exists():
        if d.parent == d:
            return False
        d = d.parent
    return os.access(d, os.W_OK)


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
