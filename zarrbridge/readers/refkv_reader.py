"""refkv reader (spec §5.4): zarrify -> ReferenceStore -> zarr store; index persisted as index.<binding> + index.json."""
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

    def open(self, path: Path, config: Any, cache_dir: Path | None = None) -> Opened:
        """`cache_dir` holds the saved index (`save_index`); any binding it was written with is read back."""
        path = Path(path).resolve()
        st = os.stat(path)
        hh = _head_hash(path, st.st_size)
        reason = "zarrified"
        rs = None
        side = _read_sidecar(cache_dir) if cache_dir is not None else None
        if side and side.get("size") == st.st_size and side.get("mtime_ns") == st.st_mtime_ns \
                and side.get("header_hash") == hh and side.get("refkv_version") == refkv.__version__ \
                and side.get("path") == str(path) and (cache_dir / side.get("file", "")).is_file():
            try:
                rs = refkv.open(str(cache_dir / side["file"]), binding=side.get("binding", "vzip"))
                reason = f"index {side['file']}"
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
        images, transcoded, classification, store = _describe(store)
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


def _read_sidecar(cache_dir: Path) -> dict | None:
    try:
        return json.loads((cache_dir / INDEX_SIDECAR).read_text())
    except (OSError, ValueError):
        return None


def has_unsaved_index(opened: Opened) -> bool:
    """True when `opened` was zarrified in this process (a saveable ReferenceStore), not reopened from an index."""
    return opened.reader_name == "refkv" and hasattr(getattr(opened, "refstore", None), "save")


def save_index(opened: Opened, cache_dir: Path, binding: str = "vzip") -> Path:
    """Persist the ReferenceStore as `cache_dir/index.<binding>` (vzip | phkv, whichever this refkv can write)
    plus `index.json` binding it to the source identity. Returns the index path. Any other `index.*` goes."""
    if not has_unsaved_index(opened):
        raise ValueError("opened has no saveable ReferenceStore (reopened from an index?)")
    rs = opened.refstore  # type: ignore[attr-defined]
    save = getattr(rs, f"save_{binding}", rs.save if binding == "vzip" else None)
    if save is None:
        raise ValueError(f"refkv {refkv.__version__} cannot write a {binding!r} index; bindings: {', '.join(refkv.BINDINGS)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / f"index.{binding}"
    tmp = cache_dir / f"index.{binding}.tmp"
    save(str(tmp))
    os.replace(tmp, index_path)
    ident = opened.identity
    side = {"path": ident.path, "size": ident.size, "mtime_ns": ident.mtime_ns, "header_hash": ident.header_hash,
            "refkv_version": refkv.__version__, "pins": ident.extras.get("pins", []), "format": ident.extras.get("format"),
            "binding": binding, "file": index_path.name}
    side_tmp = cache_dir / (INDEX_SIDECAR + ".tmp")
    side_tmp.write_text(json.dumps(side, indent=2))
    os.replace(side_tmp, cache_dir / INDEX_SIDECAR)
    # ponytail: two builders saving at once may unlink each other's .tmp; the loser logs a warning and the
    # survivor's index is equivalent. A lock around save_index if that ever shows up in logs.
    for old in cache_dir.glob("index.*"):
        if old.name not in (index_path.name, INDEX_SIDECAR):
            old.unlink(missing_ok=True)
    return index_path


def _get_json(store, key: str) -> dict | None:
    from zarr.core.buffer import default_buffer_prototype
    buf = run_sync(store.get(key, default_buffer_prototype()))
    return None if buf is None else json.loads(buf.to_bytes())


def _describe(store):
    """(images, transcoded array paths, classification, store). Bare arrays (FITS HDUs, npy) are wrapped as images."""
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
        from zarrbridge.store.bare import BareArrayStore
        bare: dict[str, dict] = {}
        for name in run_sync(_collect(store.list_dir(""))):
            if name in ("zarr.json", "refkv_source", "OME"):
                continue
            m = _get_json(store, f"{name}/zarr.json")
            if m and m.get("node_type") == "array" and len(m.get("shape", [])) >= 2:
                bare[name] = m
        if not bare:
            raise ReaderRejected("refkv hierarchy holds no OME-Zarr image or 2-D+ array")
        store = BareArrayStore(store, bare)
        image_paths = sorted(bare)
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
        translation = next((t["translation"] for t in datasets[0].get("coordinateTransformations", []) if t["type"] == "translation"), None)
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
                                native_chunks=chunks, codecs=a0.get("codecs", []), levels=levels,
                                translation=list(translation) if translation else None))
    if not images:
        raise ReaderRejected("refkv hierarchy holds no OME-Zarr image with datasets")
    return images, transcoded, classification, store


async def _collect(ait):
    return [x async for x in ait]
