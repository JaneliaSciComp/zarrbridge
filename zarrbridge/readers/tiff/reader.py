"""tiff reader (tifffile): one native segment (strip or tile) per chunk (spec §5.3)."""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from zarrbridge.readers.base import (Classification, ImageInfo, Opened, ReaderRejected, SourceIdentity,
                                     classify_chunks)
from zarrbridge.readers.mrc.fdcache import FdCache
from zarrbridge.readers.mrc.read import pread_exact
from zarrbridge.readers.tiff import meta as tmeta
from zarrbridge.store import omezarr
from zarrbridge.store.chunkstore import ChunkStore
from zarrbridge.store.codecs import OUTPUT_CODECS, encode_chunk, pad_chunk

TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
HEAD_BYTES = 64 * 1024


@dataclass
class _Level:
    """Everything a chunk GET needs for one array, captured at open time (no TiffFile on the request path)."""
    array_path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    grid: tuple[int, ...]
    n_lead: int  # leading (per-page) axes
    page_grid: tuple[int, ...]  # pages per leading axis
    chunked: tuple[int, ...]  # segments per page axis (tifffile page.chunked)
    planar: bool
    has_samples: bool  # contig samples as trailing axis
    passthrough: bool
    endian: str
    itemsize: int
    pages: list[Any]  # TiffPage | TiffFrame
    byte_length: int


class TiffReader:
    name = "tiff"
    version = 1

    def __init__(self) -> None:
        self._fds = FdCache(max_size=256)

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        return head[:4] in TIFF_MAGICS

    def open(self, path: Path, config: Any) -> Opened:
        path = Path(path)
        try:
            tf = tifffile.TiffFile(str(path))
        except Exception as e:  # tifffile raises many types
            raise ReaderRejected(f"tifffile cannot open: {e}") from e
        try:
            return self._open(path, tf, config)
        except ReaderRejected:
            tf.close()
            raise
        except Exception as e:
            tf.close()
            raise ReaderRejected(f"tifffile: {type(e).__name__}: {e}") from e

    def _open(self, path: Path, tf: tifffile.TiffFile, config: Any) -> Opened:
        st = os.stat(path)
        series = list(tf.series)
        if not series:
            raise ReaderRejected("tiff has no image series")
        multi = len(series) > 1
        metadata: dict[str, bytes] = {}
        grids: dict[str, tuple[int, ...]] = {}
        levels: dict[str, _Level] = {}
        images: list[ImageInfo] = []
        transcoded: set[str] = set()
        classification = Classification("servable")
        max_chunk = config.limits.max_chunk_bytes

        for si, s in enumerate(series):
            img_path = str(si) if multi else ""
            axes = tmeta.axis_names(s.axes)
            page0 = s.pages[0] if s.pages[0] is not None else s.keyframe
            if page0.imagedepth != 1:
                raise ReaderRejected("volumetric (ImageDepth > 1) TIFF tiles are not supported")
            sizes, unit = tmeta.physical_sizes(tf, si, page0)
            scale0 = tmeta.scale_for([a.rstrip("0123456789") for a in axes], sizes)
            datasets = []
            level_paths = []
            for li, lvl in enumerate(s.levels):
                lv = self._level(f"{img_path}/{li}" if img_path else str(li), lvl, axes, max_chunk)
                if lv is None:
                    continue
                levels[lv.array_path] = lv
                grids[lv.array_path] = lv.grid
                codecs = [{"name": "bytes", "configuration": {"endian": lv.endian}}] if lv.passthrough else OUTPUT_CODECS
                if not lv.passthrough:
                    transcoded.add(lv.array_path)
                dtype = lvl.dtype
                metadata[f"{lv.array_path}/zarr.json"] = omezarr.dumps(
                    omezarr.array_meta(lv.shape, lv.chunks, dtype, codecs, axes))
                factor = [max(1, round(s0 / s1)) for s0, s1 in zip(s.levels[0].shape, lvl.shape)]
                scale, translation = omezarr.level_transforms(scale0, factor)
                datasets.append(omezarr.dataset_entry(str(li), scale, translation))
                level_paths.append(str(li))
                if li == 0:
                    lv0 = lv
                    native_codecs = codecs
                    if lv.byte_length > max_chunk and not lv.passthrough:
                        classification = Classification("unservable", f"compressed chunk of {lv.byte_length} bytes exceeds limits.max_chunk_bytes")
                    elif classification.state == "servable":
                        classification = classify_chunks(lv.shape, lv.chunks, axes)
            if not datasets:
                raise ReaderRejected("no decodable level in series")
            ome_axes = tmeta.ome_axes(axes, unit, sizes)
            group = omezarr.multiscales_group(ome_axes, datasets, s.name or path.name,
                                              extra_attrs={"zarrbridge": {"reader": "tiff"}})
            metadata[f"{img_path}/zarr.json" if img_path else "zarr.json"] = omezarr.dumps(group)
            images.append(ImageInfo(path=img_path, shape=lv0.shape, dtype=np.dtype(s.levels[0].dtype).name, axes=ome_axes,
                                    scale=scale0, native_chunks=lv0.chunks, codecs=native_codecs, levels=level_paths,
                                    is_stack=bool(tf.is_imagej and (tf.imagej_metadata or {}).get("frames", 0) > 1
                                                  and not sizes.get("z"))))
        if multi:
            metadata["zarr.json"] = omezarr.dumps(omezarr.plain_group(
                {"zarrbridge": {"reader": "tiff", "images": [i.path for i in images]}}))

        fd = os.open(str(path), os.O_RDONLY)
        try:
            head = pread_exact(fd, min(HEAD_BYTES, st.st_size), 0)
        except BaseException:
            os.close(fd)
            raise
        offsets = [p.offset for p in tf.pages]
        hh = "sha256:" + hashlib.sha256(head + struct.pack(f"<{len(offsets)}q", *offsets)).hexdigest()
        identity = SourceIdentity(str(path), st.st_size, st.st_mtime_ns, hh,
                                  extras={"pages": len(offsets), "series": len(series), "bigtiff": tf.is_bigtiff})
        fds = self._fds

        def chunk_fn(array_path: str, coords: tuple[int, ...]) -> bytes | None:
            lv = levels.get(array_path)
            if lv is None:
                return None
            return _read_chunk(fds, path, lv, coords)

        def close() -> None:
            tf.close()
            try:
                os.close(fd)
            except OSError:
                pass

        return Opened(store=ChunkStore(metadata, chunk_fn, grids), images=images, identity=identity,
                      classification=classification, reader_name=self.name, reader_version=self.version,
                      close=close, fd=fd, transcoded=transcoded)

    @staticmethod
    def _level(array_path: str, lvl: tifffile.TiffPageSeries, axes: list[str], max_chunk: int) -> _Level | None:
        # TiffFrame objects carry their own offsets/bytecounts; decoding goes through their keyframe.
        pages = [p for p in lvl.pages if p is not None]
        if not pages:
            return None
        page0 = pages[0].keyframe
        shape = tuple(int(x) for x in lvl.shape)
        n_page = len(page0.shape)
        n_lead = len(shape) - n_page
        planar = page0.planarconfig == 2 and page0.samplesperpixel > 1
        has_samples = (not planar) and page0.samplesperpixel > 1
        if planar:
            page_chunks = (1,) + tuple(int(c) for c in page0.chunks)
        else:
            page_chunks = tuple(int(c) for c in page0.chunks)
        if len(page_chunks) != n_page:
            return None
        chunks = (1,) * n_lead + page_chunks
        page_grid = shape[:n_lead]
        if int(np.prod(page_grid)) != len(pages):
            return None
        passthrough = page0.compression == 1 and page0.predictor == 1 and page0.fillorder == 1 and page0.bitspersample in (8, 16, 32, 64)
        endian = "big" if page0.parent.byteorder == ">" else "little"
        itemsize = np.dtype(lvl.dtype).itemsize
        byte_length = int(np.prod(chunks)) * itemsize
        return _Level(array_path, shape, chunks, omezarr.grid_shape(shape, chunks), n_lead, page_grid,
                      tuple(int(c) for c in page0.chunked), planar, has_samples, passthrough, endian, itemsize,
                      pages, max(int(b) for p in pages for b in p.databytecounts) if not passthrough else byte_length)


def _read_chunk(fds: FdCache, path: Path, lv: _Level, coords: tuple[int, ...]) -> bytes | None:
    lead, within = coords[:lv.n_lead], coords[lv.n_lead:]
    page_index = int(np.ravel_multi_index(lead, lv.page_grid)) if lv.n_lead else 0
    page = lv.pages[page_index]
    keyframe = page.keyframe
    try:
        seg = int(np.ravel_multi_index(within, lv.chunked))
    except ValueError:
        return None
    offset, count = int(page.dataoffsets[seg]), int(page.databytecounts[seg])
    if count == 0:
        return b"\x00" * lv.byte_length if lv.passthrough else encode_chunk(np.zeros(lv.chunks, dtype=keyframe.dtype))
    with fds.open(path) as h:
        raw = pread_exact(h.fd, count, offset)
    if lv.passthrough:
        if len(raw) < lv.byte_length:
            raw = raw + b"\x00" * (lv.byte_length - len(raw))
        return raw
    arr = keyframe.decode(raw, seg)[0]  # (depth, h, w, samples)
    if lv.planar:
        arr = arr[0, :, :, 0][None, ...]
    elif lv.has_samples:
        arr = arr[0]
    else:
        arr = arr[0, :, :, 0]
    arr = arr.reshape((1,) * lv.n_lead + arr.shape)
    return encode_chunk(pad_chunk(arr, lv.chunks))
