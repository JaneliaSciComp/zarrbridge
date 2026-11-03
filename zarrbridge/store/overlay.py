"""PyramidOverlay (spec §6.3): levels >= 1 come from the cache directory while the fingerprint is valid."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Iterable

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import LocalStore, WrapperStore

from zarrbridge.store.keys import array_path_of, level_of


class PyramidOverlay(WrapperStore):
    def __init__(self, store: Store, cache_dir: Path, routes: Callable[[str, int], bool]) -> None:
        """`routes(image_path, level)` says whether that level is served from the cache right now."""
        super().__init__(store)
        self.cache_dir = Path(cache_dir)
        self.routes = routes
        self._local = LocalStore(str(self.cache_dir), read_only=True)

    def _with_store(self, store):
        return type(self)(store, self.cache_dir, self.routes)

    def __eq__(self, other: object) -> bool:
        return self is other

    def routes_to_cache(self, key: str) -> bool:
        ap = array_path_of(key)
        if ap is None:
            return False
        lv = level_of(ap)
        return lv is not None and lv[1] >= 1 and self.routes(lv[0], lv[1])

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        if self.routes_to_cache(key):
            try:
                return await self._local.get(key, prototype, byte_range)  # missing cached chunk -> None (404), never build
            except FileNotFoundError:
                return None
        return await self._store.get(key, prototype, byte_range)

    async def get_partial_values(self, prototype: BufferPrototype,
                                 key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        if self.routes_to_cache(key):
            return await self._local.exists(key)
        return await self._store.exists(key)

    async def getsize(self, key: str) -> int:
        if self.routes_to_cache(key):
            return await self._local.getsize(key)
        return await self._store.getsize(key)
