"""mrc reader: cubes mode (spec §5.2), port of mrc-ng-server's serving path."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from zarrbridge.readers.base import Classification, ImageInfo, Opened, ReaderRejected, SourceIdentity
from zarrbridge.readers.mrc.fdcache import FdCache
from zarrbridge.readers.mrc.header import (MRC_SUFFIXES, MrcFormatError, MrcHeader, classify_path, header_hash,
                                           parse_header, sniff_mrc)
from zarrbridge.readers.mrc.read import ChunkOutOfBounds, read_chunk
from zarrbridge.store import omezarr
from zarrbridge.store.chunkstore import ChunkStore
from zarrbridge.store.codecs import pad_chunk

AXES = ["z", "y", "x"]


class MrcReader:
    name = "mrc"
    version = 1

    def __init__(self) -> None:
        self._fds = FdCache(max_size=256)

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        return path.suffix.lower() in MRC_SUFFIXES and sniff_mrc(head)

    def open(self, path: Path, config: Any) -> Opened:
        mcfg = config.readers.mrc
        if mcfg.chunk_mode == "native":
            raise ReaderRejected("mrc chunk_mode=native is served by the refkv reader")
        path = Path(path)
        is_stack = classify_path(str(path), mcfg.stack_globs, mcfg.volume_globs)
        fd = os.open(str(path), os.O_RDONLY)
        try:
            st = os.fstat(fd)
            try:
                hdr = parse_header(fd, st.st_size, st.st_mtime_ns, assume_mode0=mcfg.assume_mode0, is_image_stack=is_stack)
            except MrcFormatError as e:
                raise ReaderRejected(f"mrc: {e}") from e
            hhash = header_hash(fd, hdr.data_offset)
        except BaseException:
            os.close(fd)
            raise

        chunk = tuple(int(c) for c in mcfg.chunk_shape)
        shape = hdr.shape_zyx
        scale = [float(v) for v in reversed(hdr.voxel_size_angstrom)]  # zyx, angstrom
        axes = [omezarr.axis(a, "angstrom") for a in AXES]
        extra_ome: dict = {}
        if hdr.dmax > hdr.dmin:
            extra_ome["omero"] = {"channels": [{"label": path.name, "color": "FFFFFF", "active": True,
                                                "window": {"min": hdr.dmin, "max": hdr.dmax,
                                                           "start": hdr.dmin, "end": hdr.dmax}}]}
        group = omezarr.multiscales_group(axes, [omezarr.dataset_entry("0", scale, None)], path.name,
                                          extra_ome=extra_ome, extra_attrs={"mrc": hdr.mirror()})
        arr = omezarr.array_meta(shape, chunk, hdr.served_dtype, omezarr.BYTES_LE, AXES)
        metadata = {"zarr.json": omezarr.dumps(group), "0/zarr.json": omezarr.dumps(arr)}
        grids = {"0": omezarr.grid_shape(shape, chunk)}
        fds = self._fds

        def chunk_fn(array_path: str, coords: tuple[int, ...]) -> bytes | None:
            kz, ky, kx = coords
            with fds.open(path) as h:
                try:
                    block = read_chunk(h.fd, hdr, kx * chunk[2], (kx + 1) * chunk[2], ky * chunk[1], (ky + 1) * chunk[1],
                                       kz * chunk[0], (kz + 1) * chunk[0])
                except ChunkOutOfBounds:
                    return None
            return pad_chunk(block, chunk).tobytes()

        def read_region(image_path: str, slices: tuple[slice, ...]) -> np.ndarray:
            (z, y, x) = slices
            with fds.open(path) as h:
                return read_chunk(h.fd, hdr, x.start or 0, x.stop if x.stop is not None else hdr.nx,
                                  y.start or 0, y.stop if y.stop is not None else hdr.ny,
                                  z.start or 0, z.stop if z.stop is not None else hdr.nz)

        def close() -> None:
            try:
                os.close(fd)
            except OSError:
                pass

        image = ImageInfo(path="", shape=shape, dtype=hdr.served_dtype.name, axes=axes, scale=scale,
                          native_chunks=chunk, codecs=omezarr.BYTES_LE, levels=["0"], is_stack=hdr.is_image_stack)
        identity = SourceIdentity(str(path), st.st_size, st.st_mtime_ns, hhash,
                                  extras={"mode": hdr.mode, "is_image_stack": hdr.is_image_stack, "chunk_shape": list(chunk)})
        return Opened(store=ChunkStore(metadata, chunk_fn, grids), images=[image], identity=identity,
                      classification=Classification("servable", "voxel_size_default: header grid/cella unusable, serving at 1 angstrom"
                                                     if hdr.voxel_size_is_default else None),
                      reader_name=self.name, reader_version=self.version,
                      close=close, fd=fd, read_region=read_region)
