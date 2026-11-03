"""MetadataStore (spec §6.4): the cached info.json is the authority for the image group zarr.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import WrapperStore

from zarrbridge.cache.layout import info_key
from zarrbridge.store.chunkstore import apply_range


class MetadataStore(WrapperStore):
    def __init__(self, store: Store, cache_dir: Path, images: list[str], is_valid: Callable[[], bool],
                 reason: Callable[[], str | None]) -> None:
        super().__init__(store)
        self.cache_dir = Path(cache_dir)
        self.images = set(images)
        self.is_valid = is_valid
        self.reason = reason

    def _with_store(self, store):
        return type(self)(store, self.cache_dir, list(self.images), self.is_valid, self.reason)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _image_of(self, key: str) -> str | None:
        if key == "zarr.json":
            return "" if "" in self.images else None
        if key.endswith("/zarr.json"):
            ip = key[: -len("/zarr.json")]
            return ip if ip in self.images else None
        return None

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        ip = self._image_of(key)
        if ip is None:
            return await self._store.get(key, prototype, byte_range)
        if self.is_valid():
            try:
                data = (self.cache_dir / info_key(ip)).read_bytes()
                json.loads(data)  # corrupt info.json -> fall through to native
                return prototype.buffer.from_bytes(apply_range(data, byte_range))
            except (OSError, ValueError):
                pass
        buf = await self._store.get(key, prototype)
        if buf is None:
            return None
        meta = json.loads(buf.to_bytes())
        zb = meta.setdefault("attributes", {}).setdefault("zarrbridge", {})
        zb["levels"] = "native"
        zb["reason"] = self.reason()
        return prototype.buffer.from_bytes(apply_range(json.dumps(meta, indent=2).encode(), byte_range))
