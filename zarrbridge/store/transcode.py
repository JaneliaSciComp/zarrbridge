"""TranscodeStore (spec §6.2): rewrite non-allowlisted codec chains to the output chain and re-encode on get."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Iterable

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.storage import WrapperStore

from zarrbridge.store.chunkstore import apply_range
import numpy as np

from zarrbridge.store.codecs import ALLOWLIST, DTYPE_MAP, OUTPUT_CODECS, decode_chunk, encode_chunk, needs_transcode
from zarrbridge.store.keys import is_metadata_key, split_chunk_key


class TranscodeStore(WrapperStore):
    def __init__(self, store: Store, allowlist: frozenset[str] = ALLOWLIST,
                 output_codecs: list[dict] | None = None, max_decodes: int | None = None,
                 dtype_map: dict[str, str] | None = None) -> None:
        super().__init__(store)
        self.allowlist = frozenset(allowlist)
        self.output_codecs = output_codecs or OUTPUT_CODECS
        self.dtype_map = dict(DTYPE_MAP if dtype_map is None else dtype_map)
        self._meta: dict[str, dict | None] = {}  # array path -> original metadata if transcoded, else None
        self._sem = asyncio.Semaphore(max_decodes or os.cpu_count() or 1)

    def _with_store(self, store):
        return type(self)(store, self.allowlist, self.output_codecs, dtype_map=self.dtype_map)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def transcoded_arrays(self) -> set[str]:
        return {k for k, v in self._meta.items() if v is not None}

    def is_transcoded(self, key: str) -> bool:
        split = split_chunk_key(key)
        ap = split[0] if split else (key[:-len("zarr.json")].rstrip("/") if is_metadata_key(key) else None)
        return ap is not None and self._meta.get(ap) is not None

    async def _array_meta(self, array_path: str) -> dict | None:
        """Original metadata if this array needs transcoding; memoised."""
        if array_path in self._meta:
            return self._meta[array_path]
        key = f"{array_path}/zarr.json" if array_path else "zarr.json"
        buf = await self._store.get(key, default_buffer_prototype())
        meta = None
        if buf is not None:
            try:
                m = json.loads(buf.to_bytes())
                meta = m if needs_transcode(m, self.allowlist, self.dtype_map) else None
            except ValueError:
                meta = None
        self._meta[array_path] = meta
        return meta

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        if is_metadata_key(key):
            ap = key[:-len("zarr.json")].rstrip("/")
            meta = await self._array_meta(ap)
            if meta is None:
                return await self._store.get(key, prototype, byte_range)
            out = dict(meta, codecs=self.output_codecs)
            target = self.dtype_map.get(meta.get("data_type"))
            if target:
                out["data_type"] = target
                try:
                    out["fill_value"] = np.dtype(target).type(meta.get("fill_value", 0)).item()
                except (TypeError, ValueError):
                    out["fill_value"] = 0
            return prototype.buffer.from_bytes(apply_range(json.dumps(out, indent=2).encode(), byte_range))
        split = split_chunk_key(key)
        if split is None:
            return await self._store.get(key, prototype, byte_range)
        meta = await self._array_meta(split[0])
        if meta is None:
            return await self._store.get(key, prototype, byte_range)
        buf = await self._store.get(key, prototype)
        if buf is None:
            return None
        async with self._sem:
            data = await asyncio.to_thread(self._transcode, buf.to_bytes(), meta)
        return prototype.buffer.from_bytes(apply_range(data, byte_range))

    def _transcode(self, data: bytes, meta: dict) -> bytes:
        arr = decode_chunk(data, meta)
        target = self.dtype_map.get(meta.get("data_type"))
        if target:
            arr = arr.astype(np.dtype(target))  # lossy for float64 -> float32; the viewer cannot render the original
        return encode_chunk(arr, self.output_codecs)

    async def get_partial_values(self, prototype: BufferPrototype,
                                 key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def getsize(self, key: str) -> int:
        if self.is_transcoded(key) or (split_chunk_key(key) and await self._array_meta(split_chunk_key(key)[0])):
            buf = await self.get(key, default_buffer_prototype())
            if buf is None:
                raise FileNotFoundError(key)
            return len(buf)
        return await self._store.getsize(key)
