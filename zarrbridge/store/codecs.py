"""Codec policy and encode/decode helpers shared by readers, TranscodeStore and the builder."""
from __future__ import annotations

import threading

import numpy as np
import zstandard
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.registry import get_pipeline_class

from zarrbridge._sync import run_sync

OUTPUT_CODECS: list[dict] = [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
]
ALLOWLIST: frozenset[str] = frozenset(
    {"bytes", "gzip", "zstd", "blosc", "crc32c", "transpose", "sharding_indexed", "numcodecs.zlib"}
)
_local = threading.local()


def _zstd() -> zstandard.ZstdCompressor:
    # ZstdCompressor objects are not thread-safe; one per thread.
    c = getattr(_local, 'zstd', None)
    if c is None:
        c = _local.zstd = zstandard.ZstdCompressor(level=3)
    return c


def codec_names(array_meta: dict) -> list[str]:
    return [c["name"] if isinstance(c, dict) else str(c) for c in array_meta.get("codecs", [])]


def needs_transcode(array_meta: dict, allowlist: frozenset[str] = ALLOWLIST) -> bool:
    return array_meta.get("node_type") == "array" and any(n not in allowlist for n in codec_names(array_meta))


def pad_chunk(arr: np.ndarray, chunk_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(arr.shape) == tuple(chunk_shape):
        return arr
    out = np.zeros(chunk_shape, dtype=arr.dtype)
    out[tuple(slice(0, n) for n in arr.shape)] = arr
    return out


def encode_chunk(arr: np.ndarray, codecs: list[dict] | None = None) -> bytes:
    """Encode a full chunk. Fast path for OUTPUT_CODECS; general path through zarr's pipeline."""
    if codecs is None or codecs == OUTPUT_CODECS:
        a = np.ascontiguousarray(arr)
        if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and not np.little_endian):
            a = a.astype(a.dtype.newbyteorder("<"))
        return _zstd().compress(a.tobytes())
    meta = _meta_for(arr.shape, arr.dtype, codecs)
    pipe = get_pipeline_class().from_codecs(meta.codecs)
    proto = default_buffer_prototype()
    spec = ArraySpec(shape=tuple(arr.shape), dtype=meta.dtype, fill_value=meta.fill_value,
                     config=ArrayConfig.from_dict({}), prototype=proto)
    nd = proto.nd_buffer.from_numpy_array(np.ascontiguousarray(arr))
    out = run_sync(pipe.encode([(nd, spec)]))
    return out[0].to_bytes()


def decode_chunk(data: bytes, array_meta: dict, chunk_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Decode one encoded chunk with the array's declared codec chain."""
    meta = ArrayV3Metadata.from_dict(array_meta)
    shape = tuple(chunk_shape) if chunk_shape else tuple(array_meta["chunk_grid"]["configuration"]["chunk_shape"])
    pipe = get_pipeline_class().from_codecs(meta.codecs)
    proto = default_buffer_prototype()
    spec = ArraySpec(shape=shape, dtype=meta.dtype, fill_value=meta.fill_value,
                     config=ArrayConfig.from_dict({}), prototype=proto)
    out = run_sync(pipe.decode([(proto.buffer.from_bytes(data), spec)]))
    nd = out[0]
    if nd is None:
        return np.full(shape, meta.fill_value, dtype=meta.dtype.to_native_dtype())
    return nd.as_numpy_array()


def _meta_for(shape, dtype, codecs) -> ArrayV3Metadata:
    return ArrayV3Metadata.from_dict({
        "zarr_format": 3, "node_type": "array", "shape": list(shape),
        "data_type": np.dtype(dtype).newbyteorder("=").name,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(shape)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0, "codecs": codecs, "attributes": {},
    })
