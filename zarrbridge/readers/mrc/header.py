"""MRC2014 header parsing (port of mrc-ng-server mrcheader.py, plus dmin/dmax/origin/exttyp).

Data on disk is C-order with x fastest: offset(x,y,z) = data_offset + (z*ny*nx + y*nx + x)*itemsize
"""
from __future__ import annotations

import dataclasses
import fnmatch
import struct
from dataclasses import dataclass

import numpy as np

from zarrbridge.readers.mrc.read import UnexpectedEOF, pread_exact

HEADER_SIZE = 1024
IMOD_STAMP = 1146047817  # ASCII "IMOD" as little-endian int32, at byte 152
IMOD_SIGNED_BIT = 0x1
_MODE_DTYPES = {1: np.dtype("<i2"), 2: np.dtype("<f4"), 6: np.dtype("<u2"), 12: np.dtype("<f2")}
# float16 widens to float32 on the wire: Neuroglancer cannot render float16.
_WIDEN_ON_SERVE = {np.dtype("<f2"): np.dtype("<f4")}
# An image stack's z axis is a slice index; cella_z carries no meaning. One unit per slice.
STACK_Z_VOXEL_ANGSTROM = 10.0
_UNSUPPORTED_MODE_REASONS = {3: "complex data", 4: "complex data"}
_VALID_MODES = {0, 1, 2, 3, 4, 6, 12}
MRC_SUFFIXES = {".mrc", ".rec", ".st", ".ali", ".map", ".mrcs"}


class MrcFormatError(Exception):
    pass


class UnsupportedModeError(MrcFormatError):
    pass


class UnsupportedByteOrderError(MrcFormatError):
    pass


class NonStandardAxisOrderError(MrcFormatError):
    pass


class TruncatedFileError(MrcFormatError):
    pass


def classify_path(relpath: str, stack_globs, volume_globs=()) -> bool:
    """Whether `relpath` names an image stack, per operator globs. Volume globs win.
    Matched against the whole path and against the basename, so `*.st` matches anywhere."""
    import os.path
    names = (relpath, os.path.basename(relpath))
    if any(fnmatch.fnmatch(n, g) for g in volume_globs for n in names):
        return False
    return any(fnmatch.fnmatch(n, g) for g in stack_globs for n in names)


def sniff_mrc(head: bytes) -> bool:
    if len(head) < 216:
        return False
    if head[208:212] == b"MAP ":
        return True
    try:
        nx, ny, nz, mode = struct.unpack_from("<4i", head, 0)
    except struct.error:
        return False
    return nx > 0 and ny > 0 and nz > 0 and mode in _VALID_MODES


@dataclass(frozen=True)
class MrcHeader:
    nx: int
    ny: int
    nz: int
    mode: int
    nxstart: int
    nystart: int
    nzstart: int
    mx: int
    my: int
    mz: int
    nsymbt: int
    mapc: int
    mapr: int
    maps: int
    cella: tuple[float, float, float]
    origin: tuple[float, float, float]
    dmin: float
    dmax: float
    dmean: float
    exttyp: str
    voxel_size_angstrom: tuple[float, float, float]
    voxel_size_is_default: bool
    is_image_stack: bool
    mode0_signedness_is_ambiguous: bool
    dtype: np.dtype
    data_offset: int
    file_size: int
    mtime_ns: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def served_dtype(self) -> np.dtype:
        return _WIDEN_ON_SERVE.get(self.dtype, self.dtype)

    def mirror(self) -> dict:
        """The header as JSON-safe attributes (spec §5.2: attributes["mrc"])."""
        d = dataclasses.asdict(self)
        d["dtype"] = self.dtype.name
        d["served_dtype"] = self.served_dtype.name
        for k in ("cella", "origin", "voxel_size_angstrom"):
            d[k] = [float(v) for v in d[k]]
        for k in ("dmin", "dmax", "dmean"):
            d[k] = float(d[k])
        d.pop("file_size"); d.pop("mtime_ns")
        return d


def _dtype_for_mode(mode: int, raw: bytes, assume_mode0: str | None) -> tuple[np.dtype, bool]:
    if mode in _UNSUPPORTED_MODE_REASONS:
        raise UnsupportedModeError(f"mode {mode} is unsupported ({_UNSUPPORTED_MODE_REASONS[mode]})")
    if mode != 0:
        if mode not in _MODE_DTYPES:
            raise UnsupportedModeError(f"mode {mode} is not a recognised MRC mode")
        return _MODE_DTYPES[mode], False
    if assume_mode0:
        return np.dtype(np.int8 if assume_mode0 == "int8" else np.uint8), False
    if int.from_bytes(raw[152:156], "little", signed=True) == IMOD_STAMP:
        flags = int.from_bytes(raw[156:160], "little", signed=True)
        return np.dtype(np.int8 if flags & IMOD_SIGNED_BIT else np.uint8), False
    return np.dtype(np.int8), True


def parse_header(fd: int, file_size: int, mtime_ns: int, assume_mode0: str | None = None,
                 is_image_stack: bool = False) -> MrcHeader:
    try:
        raw = pread_exact(fd, HEADER_SIZE, 0)
    except UnexpectedEOF as e:
        raise TruncatedFileError("file shorter than the 1024-byte MRC header") from e

    nx, ny, nz, mode = struct.unpack_from("<4i", raw, 0)
    if mode not in _VALID_MODES:
        _, _, _, be_mode = struct.unpack_from(">4i", raw, 0)
        if be_mode not in _VALID_MODES:
            raise UnsupportedByteOrderError(f"mode field ({mode}) is not a recognised value in either byte order")
        raise UnsupportedByteOrderError("file appears to be big-endian; not supported")
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise MrcFormatError(f"non-positive dimensions: nx={nx}, ny={ny}, nz={nz}")

    nxstart, nystart, nzstart = struct.unpack_from("<3i", raw, 16)
    mx, my, mz = struct.unpack_from("<3i", raw, 28)
    cella = struct.unpack_from("<3f", raw, 40)
    mapc, mapr, maps = struct.unpack_from("<3i", raw, 64)
    dmin, dmax, dmean = struct.unpack_from("<3f", raw, 76)
    (nsymbt,) = struct.unpack_from("<i", raw, 92)
    exttyp = raw[104:108].decode("ascii", "replace").strip("\x00 ")
    origin = struct.unpack_from("<3f", raw, 196)

    if (mapc, mapr, maps) != (1, 2, 3):
        raise NonStandardAxisOrderError(f"mapc,mapr,maps = {(mapc, mapr, maps)}, expected (1, 2, 3)")
    if nsymbt < 0:
        raise MrcFormatError(f"negative nsymbt: {nsymbt}")

    dtype, ambiguous = _dtype_for_mode(mode, raw, assume_mode0)
    data_offset = HEADER_SIZE + nsymbt
    required = data_offset + nx * ny * nz * dtype.itemsize
    if required > file_size:
        raise TruncatedFileError(f"file is {file_size} bytes but header implies at least {required} bytes")

    is_stack = is_image_stack
    voxel_size_is_default = False
    if mx == 0 or my == 0 or mz == 0 or all(c == 0.0 for c in cella):
        voxel_size = (1.0, 1.0, 1.0)
        voxel_size_is_default = True
    elif not is_stack and (mx, my, mz) != (nx, ny, nz):
        # mrc-ng-server refused these files. Serving at a default scale, flagged, beats not serving:
        # the pixels are fine, only the scale bar is unknown. Status carries the `voxel_size_default` advice.
        voxel_size = (1.0, 1.0, 1.0)
        voxel_size_is_default = True
    else:
        voxel_size = tuple((c / m) if c != 0.0 else 1.0 for c, m in zip(cella, (mx, my, mz)))
        voxel_size_is_default = any(c == 0.0 for c in (cella[:2] if is_stack else cella))
    if is_stack:
        voxel_size = (voxel_size[0], voxel_size[1], STACK_Z_VOXEL_ANGSTROM)

    return MrcHeader(
        nx=nx, ny=ny, nz=nz, mode=mode, nxstart=nxstart, nystart=nystart, nzstart=nzstart,
        mx=mx, my=my, mz=mz, nsymbt=nsymbt, mapc=mapc, mapr=mapr, maps=maps,
        cella=tuple(float(c) for c in cella), origin=tuple(float(o) for o in origin),
        dmin=float(dmin), dmax=float(dmax), dmean=float(dmean), exttyp=exttyp,
        voxel_size_angstrom=tuple(float(v) for v in voxel_size), voxel_size_is_default=voxel_size_is_default,
        is_image_stack=is_stack, mode0_signedness_is_ambiguous=ambiguous,
        dtype=dtype, data_offset=data_offset, file_size=file_size, mtime_ns=mtime_ns,
    )


def header_hash(fd: int, data_offset: int) -> str:
    """sha256 of the 1024-byte header plus the extended header (spec §5.2)."""
    import hashlib
    return "sha256:" + hashlib.sha256(pread_exact(fd, data_offset, 0)).hexdigest()
