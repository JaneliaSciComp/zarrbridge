"""A read-only zarr v3 Store built from a metadata dict and a chunk callback."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Iterable

from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype

from zarrbridge.store.keys import split_chunk_key

ChunkFn = Callable[[str, tuple[int, ...]], bytes | None]


def apply_range(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start:byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset:]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix:] if byte_range.suffix else b""
    raise TypeError(type(byte_range))


class ChunkStore(Store):
    supports_writes = False
    supports_deletes = False
    supports_partial_writes = False
    supports_listing = True

    def __init__(self, metadata: dict[str, bytes], chunk_fn: ChunkFn, grids: dict[str, tuple[int, ...]]):
        super().__init__(read_only=True)
        self.metadata = dict(metadata)
        self.chunk_fn = chunk_fn
        self.grids = dict(grids)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _in_grid(self, array_path: str, coords: tuple[int, ...]) -> bool:
        grid = self.grids.get(array_path)
        return grid is not None and len(grid) == len(coords) and all(0 <= c < n for c, n in zip(coords, grid))

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        if key in self.metadata:
            return prototype.buffer.from_bytes(apply_range(self.metadata[key], byte_range))
        split = split_chunk_key(key)
        if split is None or not self._in_grid(*split):
            return None
        data = await asyncio.to_thread(self.chunk_fn, *split)
        if data is None:
            return None
        return prototype.buffer.from_bytes(apply_range(data, byte_range))

    async def get_partial_values(self, prototype: BufferPrototype,
                                 key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        if key in self.metadata:
            return True
        split = split_chunk_key(key)
        return split is not None and self._in_grid(*split)

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("read-only store")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("read-only store")

    async def list(self) -> AsyncIterator[str]:
        for k in list(self.metadata):
            yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for k in list(self.metadata):
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        seen: set[str] = set()
        for k in list(self.metadata):
            if prefix and not k.startswith(prefix + "/"):
                continue
            rest = k[len(prefix) + 1:] if prefix else k
            name = rest.split("/", 1)[0]
            if name and name not in seen:
                seen.add(name)
                yield name
        if not prefix:
            return
        # chunk directories: arrays under this prefix expose a "c" entry
        if prefix in self.grids and "c" not in seen:
            yield "c"

    async def getsize(self, key: str) -> int:
        from zarr.core.buffer import default_buffer_prototype
        buf = await self.get(key, default_buffer_prototype())
        if buf is None:
            raise FileNotFoundError(key)
        return len(buf)
