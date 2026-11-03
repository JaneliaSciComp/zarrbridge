import struct

import numpy as np
import pytest
import tifffile
from zarr.core.buffer import default_buffer_prototype

from zarrbridge._sync import run_sync
from zarrbridge.server.config import Config

HEADER_SIZE = 1024
MODE_DTYPE = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}
PROTO = default_buffer_prototype()


def canonical(zz, yy, xx):
    return (xx + 1000 * yy + 1_000_000 * zz) % 30000


def make_mrc(path, shape, mode=1, voxel_size_angstrom=(1.0, 1.0, 1.0), nsymbt=0, mapc=1, mapr=2, maps=3,
             imod_flags=None, fill=None, truncate_bytes=0, nstart=(0, 0, 0), grid_size=None, dmin=0.0, dmax=0.0):
    """Port of mrc-ng-server tests/conftest.py make_mrc. shape is (nx, ny, nz)."""
    if isinstance(voxel_size_angstrom, (int, float)):
        voxel_size_angstrom = (voxel_size_angstrom,) * 3
    nx, ny, nz = shape
    dtype = MODE_DTYPE[mode]
    mx, my, mz = grid_size if grid_size is not None else (nx, ny, nz)
    cella = tuple(v * m for v, m in zip(voxel_size_angstrom, (mx, my, mz)))
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<3i", header, 0, nx, ny, nz)
    struct.pack_into("<i", header, 12, mode)
    struct.pack_into("<3i", header, 16, *nstart)
    struct.pack_into("<3i", header, 28, mx, my, mz)
    struct.pack_into("<3f", header, 40, *cella)
    struct.pack_into("<3f", header, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<3i", header, 64, mapc, mapr, maps)
    struct.pack_into("<3f", header, 76, dmin, dmax, (dmin + dmax) / 2)
    struct.pack_into("<i", header, 92, nsymbt)
    struct.pack_into("<4s", header, 104, b"    ")
    struct.pack_into("<i", header, 108, 20140)
    if imod_flags is not None:
        struct.pack_into("<i", header, 152, 1146047817)
        struct.pack_into("<i", header, 156, 1 if imod_flags == "signed" else 0)
    struct.pack_into("<4s", header, 208, b"MAP ")
    struct.pack_into("<4B", header, 212, 0x44, 0x41, 0x00, 0x00)
    if fill is None:
        data = np.zeros((nz, ny, nx), dtype=dtype)
    else:
        zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        data = fill(zz, yy, xx).astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(b"\x00" * nsymbt)
        f.write(data.tobytes())
        if truncate_bytes:
            f.truncate(f.tell() - truncate_bytes)
    return path


def make_tiff(path, arr, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr, **kwargs)
    return path


def stack_u16(shape=(5, 100, 120)):
    return np.arange(int(np.prod(shape)), dtype="<u2").reshape(shape)


def get(store, key, byte_range=None):
    buf = run_sync(store.get(key, PROTO, byte_range))
    return None if buf is None else buf.to_bytes()


@pytest.fixture
def cfg():
    return Config.load(file=None, env={})


@pytest.fixture
def cfg_min4():
    return Config.load(file=None, env={}, overrides={"build": {"min_axis": 4, "max_levels": 3, "auto": False}})
