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
from zarrbridge.cache.layout import BUILD_JSON, INDEX, cache_bytes, cache_dir_for
from zarrbridge.cache.locks import is_locked
from zarrbridge.readers.base import Opened
from zarrbridge.readers.registry import open_dataset_reader
from zarrbridge.store.keys import array_path_of, level_of
from zarrbridge.store.metadata import MetadataStore
from zarrbridge.store.overlay import PyramidOverlay
from zarrbridge.store.transcode import TranscodeStore

_log = logging.getLogger("zarrbridge.dataset")


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
        self._checked = 0.0
        self._lock = threading.Lock()
        self.opened_at = time.time()
        self.last_used = time.time()
        images = [i.path for i in opened.images]
        self.transcode = TranscodeStore(opened.store, max_decodes=config.max_decodes())
        self.overlay = PyramidOverlay(self.transcode, cache_dir, images, lambda: self.validity == "valid")
        self.store = MetadataStore(self.overlay, cache_dir, images, lambda: self.validity == "valid", self._native_reason)
        self.recheck(force=True)

    @classmethod
    def open(cls, path: Path | str, config: Any) -> "Dataset":
        path = Path(path).resolve()
        cache_dir, mode = cache_dir_for(path, config)
        opened = open_dataset_reader(path, config, index_path=cache_dir / INDEX)
        _log.info("open %s reader=%s classification=%s reason=%s", path, opened.reader_name,
                  opened.classification.state, opened.reader_reason)
        return cls(path, config, opened, cache_dir, mode)

    # -- validity -----------------------------------------------------------------------------------------------
    def recheck(self, force: bool = False) -> None:
        """Re-validate at most every cache.recheck_seconds. fstat on the held fd catches a replaced file under
        the same path even when NFS attribute caching hides it from stat()."""
        now = time.time()
        if not force and now - self._checked < self.config.cache.recheck_seconds:
            return
        with self._lock:
            self._checked = now
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
        if bj and bj.get("state") == "failed":
            return "failed", {"reason": bj.get("error")}
        reason = f"cache {self.validity}" + (f" ({self.mismatch} mismatch)" if self.mismatch else "")
        if self.config.build.auto and self.opened.identity.size > self.config.build.auto_max_bytes:
            return "queued", {"reason": f"{reason}; file exceeds build.auto_max_bytes: run `zarrbridge build` or submit a job"}
        if not self.config.build.auto:
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
        return {
            "path": str(self.path), "reader": self.opened.reader_name, "reader_reason": self.opened.reader_reason,
            "classification": self.opened.classification.state,
            "images": [{"path": i.path, "shape": list(i.shape), "dtype": i.dtype, "axes": [a["name"] for a in i.axes],
                        "native_chunks": list(i.native_chunks), "levels": i.levels} for i in self.opened.images],
            "levels": levels,
            "cache": {"mode": self.cache_mode, "dir": str(self.cache_dir),
                      "bytes": cache_bytes(self.cache_dir) if self.cache_dir.exists() else 0,
                      "fingerprint": self.validity, "mismatch": self.mismatch},
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
        if self.validity in ("missing", "stale", "outdated", "incompatible") and queue is not None:
            queue.enqueue(self.path, self.cache_dir, self.cache_mode)
        return None

    def close(self) -> None:
        self.opened.close()
