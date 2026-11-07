"""Streaming pyramid builder (spec §9). Separate process from the server; writes fingerprint.json last."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr
from zarr.storage import LocalStore

from zarrbridge.build.downsample import block_mean
from zarrbridge.build.plan import Level, choose_factor, parse_factor, plan_dict, plan_levels
from zarrbridge.cache.fingerprint import build_fingerprint, validate, write_fingerprint
from zarrbridge.cache.layout import BUILD_JSON, SOURCE_JSON, cache_dir_for, fsync_tree, info_key
from zarrbridge.cache.locks import BuildLock
from zarrbridge.readers.base import Opened, ReaderRejected
from zarrbridge.readers.registry import open_dataset_reader
from zarrbridge.store.transcode import TranscodeStore

_log = logging.getLogger("zarrbridge.build")

BUILT, ERROR, SKIPPED_LOCKED, SKIPPED_VALID, UNSERVABLE = 0, 1, 3, 4, 5
DEFAULT_MAX_BLOCK_BYTES = 256 << 20
PROGRESS_EVERY = 16


class _Progress:
    def __init__(self, cache_dir: Path, callback: Callable[[dict], None] | None = None):
        self.path = cache_dir / BUILD_JSON
        self.started = time.time()
        self.state: dict[str, Any] = {"state": "building", "level": 0, "done": 0, "total": 0,
                                      "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
                                      "eta_s": None, "bytes_read": 0, "bytes_written": 0}
        self.callback = callback
        self._n = 0

    def level(self, k: int, total: int) -> None:
        self.state.update(level=k, done=0, total=total)
        self.flush()

    def tick(self, nbytes_read: int = 0) -> None:
        self.state["done"] += 1
        self.state["bytes_read"] += nbytes_read
        self._n += 1
        if self._n % PROGRESS_EVERY == 0:
            done, total = self.state["done"], self.state["total"]
            elapsed = time.time() - self.started
            self.state["eta_s"] = round(elapsed / done * (total - done), 1) if done else None
            self.flush()

    def finish(self, **report: Any) -> dict:
        self.state.update(report)
        self.state["duration_s"] = round(time.time() - self.started, 3)
        self.flush()
        return dict(self.state)

    def flush(self) -> None:
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.state))
            os.replace(tmp, self.path)
        except OSError:
            pass
        if self.callback:
            self.callback(dict(self.state))


def build(path: Path | str, config: Any, *, factor: str | None = None, levels: int | None = None,
          min_axis: int | None = None, force: bool = False, jobs: int = 1, report: Path | str | None = None,
          cache_mode: str | None = None, progress: Callable[[dict], None] | None = None) -> int:
    """Build the pyramid cache for one file. Returns an exit code (spec §9.1)."""
    path = Path(path).resolve()
    t0 = time.time()
    try:
        os.nice(int(config.build.nice))
    except (OSError, ValueError):
        pass
    cache_dir, mode = cache_dir_for(path, config, cache_mode)
    try:
        opened = open_dataset_reader(path, config, cache_dir=cache_dir)
    except ReaderRejected as e:
        _log.error("%s: unservable: %s", path, e)
        _write_report(report, {"path": str(path), "status": "unservable", "error": str(e)})
        return UNSERVABLE
    try:
        _save_index(path, opened, config, cache_dir)  # first: the server reopens from it while levels build
        if opened.classification.state == "unservable":
            _write_report(report, {"path": str(path), "status": "unservable", "error": opened.classification.reason})
            return UNSERVABLE
        state, _ = validate(cache_dir, opened)
        if state == "valid" and not force:
            _write_report(report, {"path": str(path), "status": "skipped_valid", "cache_dir": str(cache_dir)})
            return SKIPPED_VALID
        lock = BuildLock.acquire(cache_dir, config.build.lock_stale_seconds)
        if lock is None:
            _write_report(report, {"path": str(path), "status": "skipped_locked", "cache_dir": str(cache_dir)})
            return SKIPPED_LOCKED
        try:
            result = _build_locked(path, opened, config, cache_dir, mode, factor, levels, min_axis, jobs, progress, t0)
        finally:
            lock.release()
        _write_report(report, result)
        return BUILT if result.get("state") == "done" else ERROR
    except Exception as e:  # noqa: BLE001
        _log.exception("%s: build failed", path)
        try:
            (cache_dir / BUILD_JSON).write_text(json.dumps({"state": "failed", "error": str(e)}))
        except OSError:
            pass
        _write_report(report, {"path": str(path), "status": "failed", "error": str(e)})
        return ERROR
    finally:
        opened.close()


def _save_index(path: Path, opened: Opened, config: Any, cache_dir: Path) -> None:
    """Save the refkv index when this process zarrified (none on disk, or stale). Done before validate/lock so
    a valid pyramid without an index, or a build that fails, still leaves the index behind."""
    from zarrbridge.readers.refkv_reader import has_unsaved_index, save_index
    if not has_unsaved_index(opened):
        return
    try:
        out = save_index(opened, cache_dir, config.readers.refkv.index_binding)
        _log.info("%s: saved refkv index %s", path, out)
    except Exception as e:  # noqa: BLE001
        _log.warning("%s: could not save refkv index: %s", path, e)


def _write_report(report: Path | str | None, data: dict) -> None:
    if report:
        Path(report).write_text(json.dumps(data, indent=2))


def _build_locked(path: Path, opened: Opened, config: Any, cache_dir: Path, mode: str, factor: str | None,
                  max_levels: int | None, min_axis: int | None, jobs: int,
                  progress_cb: Callable[[dict], None] | None, t0: float) -> dict:
    from zarrbridge.cache.layout import FINGERPRINT
    # Start clean: nothing half-valid may remain. Old levels and fingerprint go first.
    (cache_dir / FINGERPRINT).unlink(missing_ok=True)
    for img in opened.images:
        base = cache_dir / img.path if img.path else cache_dir
        if base.exists():
            for child in base.iterdir():
                if child.is_dir() and (child.name.isdigit() or child.name.endswith(".partial")):
                    shutil.rmtree(child)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if mode == "central":
        (cache_dir / SOURCE_JSON).write_text(json.dumps({"path": str(path)}))
    prog = _Progress(cache_dir, progress_cb)
    source_store = TranscodeStore(opened.store, max_decodes=config.max_decodes(), dtype_map=config.transcode.dtype_map)
    cache_store = LocalStore(str(cache_dir))
    plans: dict[str, list[Level]] = {}
    factors: dict[str, tuple[int, ...]] = {}
    written = 0
    for img in opened.images:
        if len(img.levels) > 1:
            # ponytail: stored pyramids (SubIFDs/OME resolutions) are served as-is; no cache levels for this image.
            _log.info("%s: image %r has %d native levels; not building", path, img.path, len(img.levels))
            continue
        f = parse_factor(factor, img) if factor and factor != "auto" else choose_factor(img, config, path)
        lv = plan_levels(img, f, min_axis if min_axis is not None else config.build.min_axis,
                         max_levels if max_levels is not None else config.build.max_levels)
        plans[img.path], factors[img.path] = lv, f
        if len(lv) < 2:
            continue
        src = zarr.open_array(source_store, path=img.level_path(0), mode="r")
        for k in range(1, len(lv)):
            prog.level(k, _n_blocks(lv[k]))
            partial = f"{img.level_path(k)}.partial"
            out = zarr.create_array(cache_store, name=partial, shape=lv[k].shape, chunks=lv[k].chunks,
                                    dtype=src.dtype, fill_value=0, dimension_names=img.axis_names,
                                    compressors=zarr.codecs.ZstdCodec(level=3, checksum=False),
                                    config={"write_empty_chunks": True}, overwrite=True)
            if k == 1:
                reader = (lambda sl, _img=img.path: opened.read_region(_img, sl)) if opened.read_region else (lambda sl: src[sl])
            else:
                prev = zarr.open_array(cache_store, path=img.level_path(k - 1), mode="r")
                reader = (lambda sl, _p=prev: _p[sl])
            step = tuple(a // b for a, b in zip(lv[k].factor, lv[k - 1].factor))
            _build_level(reader, out, lv[k], step, src.dtype.itemsize, jobs, prog)
            os.rename(cache_dir / partial, cache_dir / img.level_path(k))
        written += _dir_bytes(cache_dir / img.path if img.path else cache_dir)
        _write_info(cache_dir, img.path, source_store, lv, opened)
    fsync_tree(cache_dir)
    first = next(iter(plans.values()), None)
    plan = plan_dict(first, factors[next(iter(factors))], config.build.codec, config.build.method) if first else \
        {"factor": [], "levels": 1, "level_chunks": [], "codec": config.build.codec, "method": config.build.method}
    plan["images"] = {p: plan_dict(l, factors[p])["level_shapes"] for p, l in plans.items() if len(l) > 1}
    fp = build_fingerprint(opened, plan, duration_s=time.time() - t0)
    write_fingerprint(cache_dir, fp)
    return prog.finish(state="done", path=str(path), status="built", cache_dir=str(cache_dir),
                       bytes_written=written, levels={p: len(l) for p, l in plans.items()})


def _n_blocks(level: Level) -> int:
    n = 1
    for s, c in zip(level.shape[:-1], level.chunks[:-1]):
        n *= -(-s // c)
    return n


def _blocks(level: Level, step: tuple[int, ...], itemsize: int, max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES):
    """Output blocks: one output chunk along every axis but the last, which is full width unless the
    input block would exceed max_block_bytes (then split on output-chunk boundaries)."""
    shape, chunks = level.shape, level.chunks
    lead = [range(0, s, c) for s, c in zip(shape[:-1], chunks[:-1])]
    import itertools
    for starts in itertools.product(*lead):
        sl = [slice(s0, min(s0 + c, n)) for s0, c, n in zip(starts, chunks[:-1], shape[:-1])]
        in_voxels_per_x = itemsize
        for s, st in zip(sl, step[:-1]):
            in_voxels_per_x *= (s.stop - s.start) * st
        per_chunk = in_voxels_per_x * chunks[-1] * step[-1]
        xstep = max(1, max_block_bytes // max(per_chunk, 1)) * chunks[-1]
        for x0 in range(0, shape[-1], xstep):
            yield tuple(sl) + (slice(x0, min(x0 + xstep, shape[-1])),)


def _build_level(read: Callable[[tuple[slice, ...]], np.ndarray], out, level: Level, step: tuple[int, ...],
                 itemsize: int, jobs: int, prog: _Progress) -> None:
    def one(block: tuple[slice, ...]) -> None:
        src = tuple(slice(b.start * st, b.stop * st) for b, st in zip(block, step))
        data = read(src)
        band = block_mean(np.asarray(data), step)
        out[block] = band
        prog.tick(int(np.asarray(data).nbytes))

    blocks = list(_blocks(level, step, itemsize))
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(one, blocks))
    else:
        for b in blocks:
            one(b)


def _write_info(cache_dir: Path, image_path: str, source_store, levels: list[Level], opened: Opened) -> None:
    """info.json: the reader's group metadata with every built level appended (spec §9.3)."""
    from zarr.core.buffer import default_buffer_prototype
    from zarrbridge._sync import run_sync
    from zarrbridge.store.omezarr import dataset_entry
    key = f"{image_path}/zarr.json" if image_path else "zarr.json"
    buf = run_sync(source_store.get(key, default_buffer_prototype()))
    meta = json.loads(buf.to_bytes())
    ms = meta["attributes"]["ome"]["multiscales"][0]
    ms["datasets"] = [ms["datasets"][0]] + [dataset_entry(str(l.index), l.scale, l.translation) for l in levels[1:]]
    meta["attributes"].setdefault("zarrbridge", {}).update(
        {"levels": "multiscale" if len(levels) > 1 else "native", "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "reader": opened.reader_name})
    target = cache_dir / info_key(image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(meta, indent=2))


def _dir_bytes(d: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            total += os.lstat(os.path.join(root, f)).st_size
    return total
