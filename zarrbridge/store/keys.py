"""Zarr v3 key helpers. Keys are OME-Zarr paths relative to the dataset root."""
from __future__ import annotations


def split_chunk_key(key: str) -> tuple[str, tuple[int, ...]] | None:
    """'img/0/c/1/2/3' -> ('img/0', (1,2,3)); None for non-chunk keys."""
    parts = key.split("/")
    try:
        i = len(parts) - 1 - parts[::-1].index("c")
    except ValueError:
        return None
    if i == 0 or i == len(parts) - 1:
        return None
    try:
        coords = tuple(int(p) for p in parts[i + 1:])
    except ValueError:
        return None
    return "/".join(parts[:i]), coords


def array_path_of(key: str) -> str | None:
    if key.endswith("zarr.json"):
        p = key[: -len("zarr.json")].rstrip("/")
        return p
    s = split_chunk_key(key)
    return s[0] if s else None


def level_of(array_path: str) -> tuple[str, int] | None:
    """'img/2' -> ('img', 2); '0' -> ('', 0); None if the last segment is not an int."""
    head, _, tail = array_path.rpartition("/")
    if not tail.isdigit():
        return None
    return head, int(tail)


def is_metadata_key(key: str) -> bool:
    return key == "zarr.json" or key.endswith("/zarr.json")
