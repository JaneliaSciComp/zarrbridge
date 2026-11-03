"""BareArrayStore: present plain zarr arrays (FITS HDUs, npy, ...) as single-level OME-Zarr images.

Array `A` becomes image group `A` with dataset `0`: `A/zarr.json` is synthesized, `A/0/zarr.json` -> `A/zarr.json`,
`A/0/c/...` -> `A/c/...`. Pure key rewriting; nothing is read that was not read before."""
from __future__ import annotations

import asyncio
import json
from typing import Iterable

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import WrapperStore

from zarrbridge.store import omezarr
from zarrbridge.store.chunkstore import apply_range

_DEFAULT_AXES = ["t", "c", "z", "y", "x"]


def axes_for(meta: dict) -> list[dict]:
    names = meta.get("dimension_names")
    n = len(meta["shape"])
    # generic names (dim_0, axis1, ...) carry no meaning for a viewer: use t,c,z,y,x
    if not names or any(not x for x in names) or not any(x in ("t", "c", "z", "y", "x") for x in names):
        names = _DEFAULT_AXES[-n:] if n <= 5 else [f"dim{i}" for i in range(n)]
    return [omezarr.axis(x) if x in ("t", "c", "z", "y", "x") else {"name": x, "type": "space"} for x in names]


class BareArrayStore(WrapperStore):
    def __init__(self, store: Store, arrays: dict[str, dict]) -> None:
        """`arrays`: array path -> its zarr.json metadata, for every array to present as an image."""
        super().__init__(store)
        self.arrays = dict(arrays)
        self._groups: dict[str, bytes] = {}
        self._array_meta: dict[str, bytes] = {}
        for p, m in self.arrays.items():
            axes = axes_for(m)
            self._groups[p] = omezarr.dumps(omezarr.multiscales_group(
                axes, [omezarr.dataset_entry("0", [1.0] * len(m["shape"]), None)], p.rsplit("/", 1)[-1] or "image",
                extra_attrs={"zarrbridge": {"bare_array": True}}))
            # the array's dimension_names must agree with the multiscales axes (NGFF 0.5)
            self._array_meta[p] = omezarr.dumps(dict(m, dimension_names=[a["name"] for a in axes]))

    def _with_store(self, store):
        return type(self)(store, self.arrays)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _rewrite(self, key: str) -> str | None:
        """Inner key for a presented key, or None when the key is a synthesized group document."""
        for p in self.arrays:
            if key == f"{p}/zarr.json":
                return None
            if key == f"{p}/0/zarr.json":
                return f"{p}/zarr.json"
            if key.startswith(f"{p}/0/c/"):
                return f"{p}/c/" + key[len(p) + 5:]
        return key

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        inner = self._rewrite(key)
        if inner is None:
            return prototype.buffer.from_bytes(apply_range(self._groups[key[:-len("/zarr.json")]], byte_range))
        if inner.endswith("/zarr.json") and inner[:-len("/zarr.json")] in self._array_meta and key != inner:
            return prototype.buffer.from_bytes(apply_range(self._array_meta[inner[:-len("/zarr.json")]], byte_range))
        return await self._store.get(inner, prototype, byte_range)

    async def get_partial_values(self, prototype: BufferPrototype,
                                 key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        inner = self._rewrite(key)
        return True if inner is None else await self._store.exists(inner)

    async def getsize(self, key: str) -> int:
        inner = self._rewrite(key)
        if inner is None:
            return len(self._groups[key[:-len("/zarr.json")]])
        return await self._store.getsize(inner)

    async def list_dir(self, prefix: str):
        prefix = prefix.rstrip("/")
        if prefix in self.arrays:
            yield "zarr.json"
            yield "0"
            return
        async for k in self._store.list_dir(prefix):
            yield k
