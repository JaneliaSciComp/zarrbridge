"""HTTP header policy (spec §8.3)."""
from __future__ import annotations

from zarrbridge.dataset import Dataset
from zarrbridge.readers.mrc.read import choose_strategy
from zarrbridge.store.keys import is_metadata_key, split_chunk_key


def cache_control(ds: Dataset, key: str, immutable: bool) -> str:
    if is_metadata_key(key):
        return "no-cache"
    if immutable:
        return "public, max-age=31536000, immutable"
    if ds.is_cached_key(key):
        return "public, max-age=86400, must-revalidate"
    return "public, max-age=3600, must-revalidate"


def content_type(key: str) -> str:
    return "application/json" if is_metadata_key(key) else "application/octet-stream"


def read_strategy(ds: Dataset, key: str) -> str | None:
    if is_metadata_key(key) or ds.is_cached_key(key):
        return None
    if ds.opened.reader_name == "refkv":
        return "reference"
    if ds.opened.reader_name == "mrc":
        split = split_chunk_key(key)
        if split is None:
            return None
        img = ds.opened.images[0]
        cx = img.native_chunks[-1]
        kx = split[1][-1]
        x0, x1 = kx * cx, min((kx + 1) * cx, img.shape[-1])
        import numpy as np
        return choose_strategy(x0, x1, np.dtype(img.dtype).itemsize).value
    return "segment"
