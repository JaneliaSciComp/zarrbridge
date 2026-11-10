import json
import time

import numpy as np
import zarr

from tests.conftest import canonical, get, make_mrc
from zarrbridge.build.builder import BUILT, build
from zarrbridge.build.queue import AutoBuildQueue
from zarrbridge.dataset import Dataset
from zarrbridge.server.config import Config


def _datasets(ds):
    return [d["path"] for d in json.loads(get(ds.store, "zarr.json"))["attributes"]["ome"]["multiscales"][0]["datasets"]]


def test_native_only_then_multiscale(tmp_path, cfg_min4):
    p = make_mrc(tmp_path / "a.mrc", (36, 20, 12), fill=canonical)
    ds = Dataset.open(p, cfg_min4)
    meta = json.loads(get(ds.store, "zarr.json"))
    assert meta["attributes"]["zarrbridge"]["levels"] == "native" and "cache missing" in meta["attributes"]["zarrbridge"]["reason"]
    assert ds.status()["levels"]["state"] == "native" and ds.status()["cache"]["fingerprint"] == "missing"
    assert get(ds.store, "1/c/0/0/0") is None and ds.source_of("0/c/0/0/0") == "native"
    assert build(p, cfg_min4) == BUILT
    ds.recheck(force=True)
    assert _datasets(ds) == ["0", "1", "2"] and ds.validity == "valid"
    s = ds.status()
    assert s["levels"]["state"] == "multiscale" and s["levels"]["planned"] == 3 and s["cache"]["bytes"] > 0
    assert ds.source_of("1/c/0/0/0") == "cache" and ds.etag_for("1/c/0/0/0").startswith('W/"') and ds.etag_for("zarr.json") != ds.etag_for("0/zarr.json")
    g = zarr.open_group(ds.store, mode="r")
    assert g["1"].shape == (6, 10, 18) and g["1"][:].max() > 0
    assert get(ds.store, "1/c/5/5/5") is None  # missing cached chunk -> 404, not a build
    ds.close()


def test_replaced_file_goes_stale(tmp_path, cfg_min4):
    p = make_mrc(tmp_path / "a.mrc", (36, 20, 12), fill=canonical)
    assert build(p, cfg_min4) == BUILT
    ds = Dataset.open(p, cfg_min4)
    assert ds.validity == "valid"
    time.sleep(0.01)
    make_mrc(p, (36, 20, 12), fill=lambda zz, yy, xx: xx)  # same size, new bytes
    ds.recheck(force=True)
    assert ds.validity == "stale" and ds.mismatch in ("mtime_ns", "header_hash") and ds.source_changed()
    assert _datasets(ds) == ["0"] and ds.status()["levels"]["state"] == "native"
    assert get(ds.store, "1/c/0/0/0") is None


def test_queue_dedupes_and_spawns(tmp_path):
    calls = []

    class FakeProc:
        returncode = 0

        def communicate(self):
            return "", ""

    def popen(argv, **kw):
        calls.append(argv)
        return FakeProc()

    cfg = Config.load(env={}, overrides={"build": {"nice": 0}})
    q = AutoBuildQueue(cfg, popen=popen)
    assert q.enqueue(tmp_path / "a.mrc", tmp_path / "c", "beside") is True
    assert q.enqueue(tmp_path / "a.mrc", tmp_path / "c", "beside") is False
    deadline = time.time() + 5
    while time.time() < deadline and q.pending(tmp_path / "c"):
        time.sleep(0.01)
    assert len(calls) == 1 and calls[0][-4:] == ["--cache-mode", "beside", "build", str(tmp_path / "a.mrc")]
    from zarrbridge.cli import build_parser
    ns = build_parser().parse_args(calls[0][3:])  # the spawned argv must parse with the real CLI
    assert ns.cmd == "build" and ns.cache_mode == "beside"
    assert q.snapshot()["finished"][str(tmp_path / "c")] == 0
    q.shutdown()


def test_ensure_pyramid_auto_respects_max_bytes(tmp_path):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    cfg = Config.load(env={}, overrides={"build": {"auto_max_bytes": 10}})
    ds = Dataset.open(p, cfg)
    q = AutoBuildQueue(cfg, popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    ds.ensure_pyramid("auto", q)
    st = ds.status(q)
    # Nothing is queued, so the state must be terminal: a client polls on "queued".
    assert st["levels"]["state"] == "native" and "zarrbridge build" in st["levels"]["reason"]
    cfg2 = Config.load(env={}, overrides={"build": {"min_axis": 4, "max_levels": 2}})
    ds2 = Dataset.open(p, cfg2)
    assert ds2.ensure_pyramid("now") == BUILT and ds2.validity == "valid"


def test_auto_build_crash_is_reported_as_failed(tmp_path):
    # The builder writes build.json {"state": "failed"} itself when it can; a
    # subprocess that dies before that (OOM kill, spawn failure) must still end
    # in "failed" rather than quietly dropping back to "native".
    class DeadProc:
        returncode = 137

        def communicate(self):
            return "", "Killed"

    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    cfg = Config.load(env={}, overrides={"build": {"nice": 0}})
    ds = Dataset.open(p, cfg)
    q = AutoBuildQueue(cfg, popen=lambda *a, **k: DeadProc())
    assert q.failure(ds.cache_dir) is None
    ds.ensure_pyramid("auto", q)
    deadline = time.time() + 5
    while time.time() < deadline and q.pending(ds.cache_dir):
        time.sleep(0.01)
    assert q.failure(ds.cache_dir) == 137
    st = ds.status(q)
    assert st["levels"]["state"] == "failed" and "137" in st["levels"]["reason"]
    q.shutdown()
