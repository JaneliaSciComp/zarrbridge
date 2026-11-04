"""Fingerprint compute/write/validate (spec §7.3). fingerprint.json is written LAST; its presence means complete.

Two version numbers answer different questions:
  SCHEMA_VERSION      the shape of this JSON (keys/types). Mismatch -> incompatible.
  DERIVATION_VERSION  the meaning of the chunks beside it (downsampling method, padding, translation
                      convention, chunk-key encoding). Mismatch -> outdated.
Only ever increment. Changelog:
  derivation 1: initial. block mean, half-away-from-zero rounding, zero padding, s0*(f-1)/2 translation.
  derivation 2: level translations add the level-0 translation (t0 + s0*(f-1)/2); v1 dropped t0 for images
                with a stage offset, so coarse levels rendered displaced from level 0.
  derivation 3: float64/float16 sources build float32 levels (transcode.dtype_map); viewers have no float64.
                Changing transcode.dtype_map in config needs `zarrbridge build --force`.
  derivation 4: downsampled axes halve together; the pyramid stops when every downsampled axis would fall below
                min_axis. v3 stopped axes individually, leaving an anisotropic last level that made Neuroglancer
                render only that level once its multi-channel setup reordered the display dimensions.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Literal

import refkv

from zarrbridge import __version__
from zarrbridge.cache.layout import FINGERPRINT
from zarrbridge.cache.locks import is_locked
from zarrbridge.readers.base import Opened

SCHEMA_VERSION = 1
DERIVATION_VERSION = 4

Validity = Literal["valid", "stale", "outdated", "incompatible", "missing", "building"]


def build_fingerprint(opened: Opened, plan: dict, built_by: str | None = None, duration_s: float = 0.0) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "zarrbridge_version": __version__,
        "reader": {"name": opened.reader_name, "version": opened.reader_version},
        "refkv_version": refkv.__version__,
        "source": opened.identity.to_dict(),
        "plan": plan,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "built_by": built_by or f"{os.environ.get('USER', 'unknown')}@{socket.gethostname()}",
        "duration_s": round(duration_s, 3),
    }


def write_fingerprint(cache_dir: Path, fp: dict) -> None:
    cache_dir = Path(cache_dir)
    path = cache_dir / FINGERPRINT
    tmp = cache_dir / (FINGERPRINT + ".tmp")
    with open(tmp, "w") as f:
        json.dump(fp, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(cache_dir, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def read_fingerprint(cache_dir: Path) -> dict | None:
    try:
        with open(Path(cache_dir) / FINGERPRINT) as f:
            fp = json.load(f)
    except (OSError, ValueError):
        return None
    return fp if isinstance(fp, dict) else None


def fingerprint_digest(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


def validate(cache_dir: Path, opened: Opened) -> tuple[Validity, str | None]:
    """(validity, mismatching field). Only `valid` serves levels >= 1."""
    fp = read_fingerprint(cache_dir)
    if fp is None:
        return ("building" if is_locked(cache_dir) else "missing"), None
    if fp.get("schema_version") != SCHEMA_VERSION:
        return "incompatible", "schema_version"
    if fp.get("derivation_version") != DERIVATION_VERSION:
        return "outdated", "derivation_version"
    rd = fp.get("reader", {})
    if rd.get("name") != opened.reader_name:
        return "outdated", "reader.name"
    if rd.get("version") != opened.reader_version:
        return "outdated", "reader.version"
    if opened.reader_name == "refkv" and fp.get("refkv_version") != refkv.__version__:
        return "outdated", "refkv_version"
    src = fp.get("source", {})
    cur = opened.identity.to_dict()
    for field in ("path", "size", "mtime_ns", "header_hash"):
        if src.get(field) != cur.get(field):
            return "stale", field
    for field in (k for k in cur if k not in ("path", "size", "mtime_ns", "header_hash")):
        if src.get(field) != cur.get(field):
            return "stale", field
    return "valid", None
