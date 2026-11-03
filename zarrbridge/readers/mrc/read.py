"""pread-based chunk extraction (port of mrc-ng-server reader.py). Every MRC read goes through pread_exact."""
from __future__ import annotations

import enum
import os

import numpy as np

# Test instrumentation: set ZB_INSTRUMENT_READS=1 and read READ_COUNTER = [calls, bytes].
READ_COUNTER = [0, 0]
_INSTRUMENT = bool(os.environ.get("ZB_INSTRUMENT_READS"))


class UnexpectedEOF(Exception):
    pass


class ChunkOutOfBounds(Exception):
    pass


class ReadStrategy(enum.Enum):
    ROW_WISE = "rowwise"
    SPAN_WISE = "spanwise"


def pread_exact(fd: int, count: int, offset: int) -> bytes:
    chunks = []
    got = 0
    while got < count:
        chunk = os.pread(fd, count - got, offset + got)
        if not chunk:
            raise UnexpectedEOF(f"unexpected EOF: got {got} of {count} bytes at offset {offset}")
        chunks.append(chunk)
        got += len(chunk)
    if _INSTRUMENT:
        READ_COUNTER[0] += 1
        READ_COUNTER[1] += count
    return b"".join(chunks)


def choose_strategy(x0: int, x1: int, itemsize: int, threshold: int = 4096) -> ReadStrategy:
    row_bytes = (x1 - x0) * itemsize
    return ReadStrategy.ROW_WISE if row_bytes >= threshold else ReadStrategy.SPAN_WISE


def read_chunk(fd: int, hdr, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
               row_bytes_threshold: int = 4096) -> np.ndarray:
    """Clipped (z,y,x) block in hdr.served_dtype. Never padded."""
    x0 = max(x0, 0); y0 = max(y0, 0); z0 = max(z0, 0)
    x1 = min(x1, hdr.nx); y1 = min(y1, hdr.ny); z1 = min(z1, hdr.nz)
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        raise ChunkOutOfBounds(f"empty clipped region: x[{x0}:{x1}] y[{y0}:{y1}] z[{z0}:{z1}]")

    itemsize = hdr.dtype.itemsize
    strategy = choose_strategy(x0, x1, itemsize, row_bytes_threshold)
    out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=hdr.served_dtype)
    row_len = x1 - x0
    if strategy is ReadStrategy.ROW_WISE:
        for zi, z in enumerate(range(z0, z1)):
            for yi, y in enumerate(range(y0, y1)):
                offset = hdr.data_offset + (z * hdr.ny * hdr.nx + y * hdr.nx + x0) * itemsize
                raw = pread_exact(fd, row_len * itemsize, offset)
                out[zi, yi, :] = np.frombuffer(raw, dtype=hdr.dtype, count=row_len)
    else:
        # One pread per z spanning (y0,x0)..(y1-1,x1): (y1-y0)x fewer syscalls for an nx/(x1-x0) over-read.
        span_len = (y1 - 1 - y0) * hdr.nx + row_len
        for zi, z in enumerate(range(z0, z1)):
            offset = hdr.data_offset + (z * hdr.ny * hdr.nx + y0 * hdr.nx + x0) * itemsize
            raw = pread_exact(fd, span_len * itemsize, offset)
            span = np.frombuffer(raw, dtype=hdr.dtype, count=span_len)
            for yi in range(y1 - y0):
                out[zi, yi, :] = span[yi * hdr.nx: yi * hdr.nx + row_len]
    return out
