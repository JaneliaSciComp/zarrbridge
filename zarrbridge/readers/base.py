"""Reader contracts (spec §5.1)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
from zarr.abc.store import Store


class ReaderRejected(Exception):
    """A reader cannot serve this file; the message is the reason (spec §3.7)."""


class IndexTimeout(ReaderRejected):
    """The reader did not finish indexing within the server's deadline. Terminal: the file must be
    indexed by hand (`zarrbridge index`), which has no deadline."""

    def __init__(self, path, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__(f"Indexing did not finish within {timeout_s:.0f} s, so this file is too big to index "
                         f"automatically. It needs to be indexed manually with `zarrbridge index {path}`.")


@dataclass(frozen=True)
class ImageInfo:
    path: str  # group path relative to dataset root; "" when the image is the root
    shape: tuple[int, ...]
    dtype: str
    axes: list[dict]
    scale: list[float]
    native_chunks: tuple[int, ...]
    codecs: list[dict]
    levels: list[str]  # dataset paths present natively ("0", or "0","1",... for stored pyramids)
    is_stack: bool = False
    translation: list[float] | None = None  # level-0 translation; built levels add their half-pixel offset to it

    def level_path(self, k: int | str) -> str:
        return f"{self.path}/{k}" if self.path else str(k)

    @property
    def axis_names(self) -> list[str]:
        return [a["name"] for a in self.axes]


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    size: int
    mtime_ns: int
    header_hash: str
    extras: dict = field(default_factory=dict)

    def digest(self) -> str:
        blob = json.dumps({"size": self.size, "header_hash": self.header_hash, "extras": self.extras},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns,
                "header_hash": self.header_hash, **self.extras}


@dataclass(frozen=True)
class Classification:
    state: Literal["servable", "servable_2d", "unservable"]
    reason: str | None = None


@dataclass
class Opened:
    store: Store
    images: list[ImageInfo]
    identity: SourceIdentity
    classification: Classification
    reader_name: str
    reader_version: int
    reader_reason: str = ""
    close: Callable[[], None] = lambda: None
    fd: int | None = None
    # Optional builder fast path: read_region(image_path, slices) -> ndarray at level 0
    read_region: Callable[[str, tuple[slice, ...]], np.ndarray] | None = None
    # array paths whose chunks are re-encoded on the request path
    transcoded: set[str] = field(default_factory=set)


class Reader(Protocol):
    name: str
    version: int

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool: ...

    def open(self, path: Path, config: Any) -> Opened: ...


def sha256_prefix(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def classify_chunks(shape: tuple[int, ...], chunks: tuple[int, ...], axis_names: list[str]) -> Classification:
    """servable_2d when a spatial axis of extent > 1 is chunked at 1 (plane-chunked stacks)."""
    for n, c, name in zip(shape, chunks, axis_names):
        if name in ("z",) and n > 1 and c == 1:
            return Classification("servable_2d")
    return Classification("servable")
