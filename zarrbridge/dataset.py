"""Dataset: one source file plus its derived state (spec §2, §8.4, §12)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from zarrbridge.build.queue import AutoBuildQueue
from zarrbridge.cache.fingerprint import Validity, fingerprint_digest, read_fingerprint, validate
from zarrbridge.cache.layout import BUILD_JSON, FINGERPRINT, cache_bytes, cache_dir_for, cache_writable
from zarrbridge.cache.locks import is_locked
from zarrbridge.readers.base import Opened
from zarrbridge.readers.registry import open_dataset_reader
from zarrbridge.store.keys import array_path_of, level_of
from zarrbridge.store.metadata import MetadataStore
from zarrbridge.store.overlay import PyramidOverlay
from zarrbridge.store.transcode import TranscodeStore

_log = logging.getLogger("zarrbridge.dataset")


def _save_index_if_slow(path: Path, opened: Opened, config: Any, cache_dir: Path, elapsed: float) -> None:
    """A server-side zarrify that took a while is paid once: persist its refkv index beside the cache so the
    next open reads the index instead of the file. Needs a writable cache; failures only log."""
    from zarrbridge.readers.refkv_reader import has_unsaved_index, save_index
    if elapsed < config.limits.index_save_after_s or not has_unsaved_index(opened) or not cache_writable(cache_dir):
        return
    try:
        out = save_index(opened, cache_dir, config.readers.refkv.index_binding)
        _log.info("%s: saved refkv index %s after a %.1fs open", path, out, elapsed)
    except Exception as e:  # noqa: BLE001
        _log.warning("%s: could not save refkv index: %s", path, e)


class Dataset:
    def __init__(self, path: Path, config: Any, opened: Opened, cache_dir: Path, cache_mode: str) -> None:
        self.path = Path(path)
        self.config = config
        self.opened = opened
        self.cache_dir = cache_dir
        self.cache_mode = cache_mode
        self.validity: Validity = "missing"
        self.mismatch: str | None = None
        self._fingerprint: dict | None = None
        self._fp_digest = ""
        self._fp_key: tuple | None = None
        self._checked = 0.0
        self._lock = threading.Lock()
        self.opened_at = time.time()
        self.last_used = time.time()
        images = [i.path for i in opened.images]
        self.transcode = TranscodeStore(opened.store, max_decodes=config.max_decodes(), dtype_map=config.transcode.dtype_map)
        self._native_levels = {i.path: len(i.levels) for i in opened.images}
        self.overlay = PyramidOverlay(self.transcode, cache_dir, self._routes_to_cache)
        self.store = MetadataStore(self.overlay, cache_dir, images, lambda: self.validity == "valid", self._native_reason)
        self.recheck(force=True)

    @classmethod
    def open(cls, path: Path | str, config: Any) -> "Dataset":
        path = Path(path).resolve()
        cache_dir, mode = cache_dir_for(path, config)
        t0 = time.time()
        opened = open_dataset_reader(path, config, cache_dir=cache_dir, timeout=config.limits.open_timeout_s or None)
        elapsed = time.time() - t0
        _log.info("open %s reader=%s classification=%s reason=%s in %.1fs", path, opened.reader_name,
                  opened.classification.state, opened.reader_reason, elapsed)
        _save_index_if_slow(path, opened, config, cache_dir, elapsed)
        return cls(path, config, opened, cache_dir, mode)

    # -- validity -----------------------------------------------------------------------------------------------
    def recheck(self, force: bool = False) -> None:
        """Re-validate at most every cache.recheck_seconds. fstat on the held fd catches a replaced file under
        the same path even when NFS attribute caching hides it from stat()."""
        now = time.time()
        fp_key = self._fingerprint_key()
        if not force and fp_key == self._fp_key and now - self._checked < self.config.cache.recheck_seconds:
            return
        with self._lock:
            self._checked = now
            self._fp_key = fp_key
            ident = self.opened.identity
            if self.opened.fd is not None:
                try:
                    st = os.fstat(self.opened.fd)
                    if st.st_size != ident.size or st.st_mtime_ns != ident.mtime_ns:
                        self.validity, self.mismatch = "stale", "size" if st.st_size != ident.size else "mtime_ns"
                        return
                except OSError:
                    pass
            try:
                st = os.stat(self.path)
                if st.st_size != ident.size or st.st_mtime_ns != ident.mtime_ns:
                    self.validity, self.mismatch = "stale", "size" if st.st_size != ident.size else "mtime_ns"
                    return
            except OSError:
                self.validity, self.mismatch = "stale", "path"
                return
            self.validity, self.mismatch = validate(self.cache_dir, self.opened)
            if self.validity == "valid":
                self._fingerprint = read_fingerprint(self.cache_dir)
                self._fp_digest = fingerprint_digest(self._fingerprint) if self._fingerprint else ""
            else:
                self._fingerprint, self._fp_digest = None, ""

    def _fingerprint_key(self) -> tuple | None:
        # one stat per request: a build that completes is picked up on the next request, not after recheck_seconds
        try:
            st = os.stat(self.cache_dir / FINGERPRINT)
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    def _routes_to_cache(self, image_path: str, level: int) -> bool:
        """Only levels the builder wrote: images with a stored (native) pyramid are never overlaid."""
        if self.validity != "valid" or self._fingerprint is None:
            return False
        built = self._fingerprint.get("plan", {}).get("images", {})
        return image_path in built and level >= self._native_levels.get(image_path, 1)

    def source_changed(self) -> bool:
        """True when the file on disk no longer matches the identity we opened (reopen needed)."""
        ident = self.opened.identity
        try:
            st = os.stat(self.path)
        except OSError:
            return True
        return st.st_size != ident.size or st.st_mtime_ns != ident.mtime_ns

    def _native_reason(self) -> str | None:
        return self.status()["levels"]["reason"]

    # -- per-key facts for HTTP ----------------------------------------------------------------------------------
    def is_cached_key(self, key: str) -> bool:
        return self.overlay.routes_to_cache(key)

    def etag_for(self, key: str) -> str:
        if self.is_cached_key(key):
            return f'W/"{self._fp_digest}:{key}"'
        tag = self.opened.identity.digest()
        if key.endswith("zarr.json") and self.validity == "valid":
            tag = f"{tag}+{self._fp_digest}"
        return f'W/"{tag}:{key}"'

    def source_of(self, key: str) -> str:
        if self.is_cached_key(key):
            return "cache"
        return "transcoded" if self.transcode.is_transcoded(key) or (array_path_of(key) or "") in self.opened.transcoded else "native"

    # -- status (§8.4) -----------------------------------------------------------------------------------------
    def levels_state(self, queue: AutoBuildQueue | None = None) -> tuple[str, dict]:
        cls = self.opened.classification
        if cls.state == "unservable":
            return "unservable", {"reason": cls.reason}
        if self.validity == "valid":
            plan = (self._fingerprint or {}).get("plan", {})
            if not plan.get("images"):
                return "native", {"reason": "no level above native in the plan (image smaller than build.min_axis)"}
            return "multiscale", {"reason": None}
        bj = self._build_json()
        if is_locked(self.cache_dir):
            st = {"reason": None}
            if bj and bj.get("state") == "building":
                done, total = bj.get("done", 0), bj.get("total", 0)
                st.update(built=bj.get("level", 0) - 1, progress=round(done / total, 3) if total else 0.0, eta_s=bj.get("eta_s"))
            return ("indexing" if bj is None else "building"), st
        if queue is not None and queue.pending(self.cache_dir):
            return "queued", {"reason": "auto-build queued"}
        # Terminal states from here on: indexing/queued/building are the only
        # ones a client should poll, so a build that will not happen or did
        # not succeed must never be reported as one of them.
        if bj and bj.get("state") == "failed":
            return "failed", {"reason": bj.get("error")}
        rc = queue.failure(self.cache_dir) if queue is not None else None
        if rc is not None:
            return "failed", {"reason": f"auto-build exited {rc} without writing a result"}
        reason = f"cache {self.validity}" + (f" ({self.mismatch} mismatch)" if self.mismatch else "")
        if not cache_writable(self.cache_dir):
            reason += "; cache directory not writable by this user: its owner must run `zarrbridge build`"
        elif self.config.build.auto and self.opened.identity.size > self.config.build.auto_max_bytes:
            reason += "; file exceeds build.auto_max_bytes: run `zarrbridge build` or submit a job"
        elif not self.config.build.auto:
            reason += "; auto-build off: run `zarrbridge build`"
        return "native", {"reason": reason}

    def _build_json(self) -> dict | None:
        try:
            return json.loads((self.cache_dir / BUILD_JSON).read_text())
        except (OSError, ValueError):
            return None

    def status(self, queue: AutoBuildQueue | None = None) -> dict:
        state, extra = self.levels_state(queue)
        planned = None
        fp = self._fingerprint or read_fingerprint(self.cache_dir)
        if fp:
            planned = fp.get("plan", {}).get("levels")
        levels = {"state": state, "built": extra.get("built", planned if state == "multiscale" else 0),
                  "planned": planned, "progress": extra.get("progress", 1.0 if state == "multiscale" else 0.0),
                  "eta_s": extra.get("eta_s"), "reason": extra.get("reason")}
        advice = ["xy_only"] if self.opened.classification.state == "servable_2d" else []
        if self.opened.classification.state != "unservable" and self.opened.classification.reason:
            advice.append(self.opened.classification.reason.split(":")[0])
        return {
            "path": str(self.path), "reader": self.opened.reader_name, "reader_reason": self.opened.reader_reason,
            "classification": self.opened.classification.state,
            "images": [{"path": i.path, "shape": list(i.shape), "dtype": i.dtype, "axes": [a["name"] for a in i.axes],
                        "native_chunks": list(i.native_chunks), "levels": i.levels} for i in self.opened.images],
            "levels": levels,
            "cache": {"mode": self.cache_mode, "dir": str(self.cache_dir),
                      "bytes": cache_bytes(self.cache_dir) if self.cache_dir.exists() else 0,
                      "fingerprint": self.validity, "mismatch": self.mismatch,
                      "writable": cache_writable(self.cache_dir)},
            "advice": advice,
        }

    # -- building (§12) ---------------------------------------------------------------------------------------
    def ensure_pyramid(self, mode: Literal["never", "auto", "now"] = "auto", queue: AutoBuildQueue | None = None) -> int | None:
        if mode == "never" or self.validity == "valid" or self.opened.classification.state == "unservable":
            return None
        if mode == "now":
            from zarrbridge.build.builder import build
            rc = build(self.path, self.config, cache_mode=self.cache_mode)
            self.recheck(force=True)
            return rc
        if not self.config.build.auto or self.opened.identity.size > self.config.build.auto_max_bytes:
            return None
        if not cache_writable(self.cache_dir):
            return None  # another user's beside cache gone stale: a build here can only fail
        if self.validity in ("missing", "stale", "outdated", "incompatible") and queue is not None:
            queue.enqueue(self.path, self.cache_dir, self.cache_mode)
        return None

    def close(self) -> None:
        self.opened.close()
