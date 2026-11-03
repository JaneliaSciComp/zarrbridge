"""refkv reader (spec §5.4): zarrify -> ReferenceStore -> zarr store; index persisted as index.vzip."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import refkv
import refkv.ir
import refkv.zarr as rzarr  # registers refkv codecs with zarr

from zarrbridge._sync import run_sync
from zarrbridge.readers.base import (Classification, ImageInfo, Opened, ReaderRejected, SourceIdentity,
                                     classify_chunks)
from zarrbridge.store.codecs import ALLOWLIST, codec_names, needs_transcode

HEAD_BYTES = 64 * 1024
INDEX_SIDECAR = "index.json"


def _head_hash(path: Path, size: int) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read(min(HEAD_BYTES, size))).hexdigest()


class RefkvReader:
    name = "refkv"
    version = 1

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool:
        try:
            return refkv.ir.sniff(head[:552]) is not None
        except Exception:
            return False

    def open(self, path: Path, config: Any, index_path: Path | None = None) -> Opened:
        path = Path(path).resolve()
        st = os.stat(path)
        hh = _head_hash(path, st.st_size)
        reason = "zarrified"
        rs = None
        if index_path is not None and index_path.exists():
            side = _read_sidecar(index_path)
            if side and side.get("size") == st.st_size and side.get("mtime_ns") == st.st_mtime_ns \
                    and side.get("header_hash") == hh and side.get("refkv_version") == refkv.__version__ \
                    and side.get("path") == str(path):
                try:
                    rs = refkv.open(str(index_path))
                    reason = f"index {index_path.name}"
                    pins = side.get("pins", [])
                    fmt = side.get("format")
                except Exception:
                    rs = None
        if rs is None:
            try:
                rs = _zarrify(path, config)
            except (refkv.zarrify.Rejected, NotImplementedError, ValueError) as e:
                raise ReaderRejected(f"refkv rejected: {e}") from e
            pins = [{"url": s.url, "size": s.size, "etag": s.etag, "modified_not_after": s.modified_not_after}
                    for s in rs.sources if s.url]
            fmt = rs.info.get("format") if isinstance(rs.info, dict) else None
            for pin in pins:
                if pin["url"] in (str(path), path.as_uri()) and pin["size"] not in (None, st.st_size):
                    raise ReaderRejected("refkv pin mismatch: size")
        store = rzarr.store(rs)
        images, transcoded, classification = _describe(store)
        identity = SourceIdentity(str(path), st.st_size, st.st_mtime_ns, hh,
                                  extras={"pins": pins, "format": fmt})
        fd = os.open(str(path), os.O_RDONLY)

        def close() -> None:
            try:
                os.close(fd)
            except OSError:
                pass

        op = Opened(store=store, images=images, identity=identity, classification=classification,
                    reader_name=self.name, reader_version=self.version, reader_reason=reason, close=close, fd=fd,
                    transcoded=transcoded)
        op.refstore = rs  # type: ignore[attr-defined]
        return op


def _zarrify(path: Path, config: Any):
    from refkv.zarrify import zarrify
    return zarrify(str(path), connections=config.readers.refkv.connections)


def _read_sidecar(index_path: Path) -> dict | None:
    try:
        return json.loads((index_path.parent / INDEX_SIDECAR).read_text())
    except (OSError, ValueError):
        return None


def save_index(opened: Opened, index_path: Path) -> None:
    """Persist the ReferenceStore as a vzip plus a sidecar binding it to the source identity."""
    rs = getattr(opened, "refstore", None)
    if rs is None or not hasattr(rs, "save"):
        raise ValueError("opened has no saveable ReferenceStore (reopened from an index?)")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".vzip.tmp")
    save = getattr(rs, "save_vzip", None) or rs.save
    save(str(tmp))
    os.replace(tmp, index_path)
    ident = opened.identity
    side = {"path": ident.path, "size": ident.size, "mtime_ns": ident.mtime_ns, "header_hash": ident.header_hash,
            "refkv_version": refkv.__version__, "pins": ident.extras.get("pins", []), "format": ident.extras.get("format")}
    (index_path.parent / INDEX_SIDECAR).write_text(json.dumps(side, indent=2))


def _get_json(store, key: str) -> dict | None:
    from zarr.core.buffer import default_buffer_prototype
    buf = run_sync(store.get(key, default_buffer_prototype()))
    return None if buf is None else json.loads(buf.to_bytes())


def _describe(store) -> tuple[list[ImageInfo], set[str], Classification]:
    root = _get_json(store, "zarr.json")
    if root is None:
        raise ReaderRejected("refkv store has no root zarr.json")
    ome = root.get("attributes", {}).get("ome", {})
    image_paths: list[str] = []
    if "multiscales" in ome:
        image_paths = [""]
    elif "bioformats2raw.layout" in ome:
        series = (_get_json(store, "OME/zarr.json") or {}).get("attributes", {}).get("ome", {}).get("series")
        if series:
            image_paths = list(series)
        else:
            image_paths = [k for k in run_sync(_collect(store.list_dir(""))) if k.isdigit()]
    if not image_paths:
        raise ReaderRejected("refkv hierarchy holds no OME-Zarr image")
    images: list[ImageInfo] = []
    transcoded: set[str] = set()
    classification = Classification("servable")
    for ip in image_paths:
        g = _get_json(store, f"{ip}/zarr.json" if ip else "zarr.json") or {}
        ms = g.get("attributes", {}).get("ome", {}).get("multiscales", [{}])[0]
        axes = ms.get("axes", [])
        datasets = ms.get("datasets", [])
        if not datasets:
            continue
        levels = [d["path"] for d in datasets]
        scale = next((t["scale"] for t in datasets[0].get("coordinateTransformations", []) if t["type"] == "scale"),
                     [1.0] * len(axes))
        for lp in levels:
            ap = f"{ip}/{lp}" if ip else lp
            am = _get_json(store, f"{ap}/zarr.json")
            if am and needs_transcode(am):
                transcoded.add(ap)
        a0 = _get_json(store, f"{(ip + '/') if ip else ''}{levels[0]}/zarr.json") or {}
        shape = tuple(a0.get("shape", []))
        chunks = tuple(a0.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape", shape))
        names = [a["name"] for a in axes]
        if classification.state == "servable":
            classification = classify_chunks(shape, chunks, names)
        images.append(ImageInfo(path=ip, shape=shape, dtype=a0.get("data_type", ""), axes=axes, scale=list(scale),
                                native_chunks=chunks, codecs=a0.get("codecs", []), levels=levels))
    if not images:
        raise ReaderRejected("refkv hierarchy holds no OME-Zarr image with datasets")
    return images, transcoded, classification


async def _collect(ait):
    return [x async for x in ait]
