import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from tests.conftest import canonical, make_mrc, make_tiff, stack_u16
from zarrbridge.build.builder import BUILT, ERROR, SKIPPED_LOCKED, SKIPPED_VALID, UNSERVABLE, build
from zarrbridge.build.downsample import block_mean
from zarrbridge.cache.fingerprint import validate
from zarrbridge.cache.locks import BuildLock
from zarrbridge.readers.registry import open_dataset_reader


def test_build_mrc_parity_with_reference(tmp_path, cfg_min4):
    p = make_mrc(tmp_path / "a.mrc", (36, 20, 12), fill=canonical)
    assert build(p, cfg_min4) == BUILT
    cache = tmp_path / ".zarrbridge" / "a.mrc"
    assert (cache / "fingerprint.json").exists() and not list(cache.glob("*.partial")) and not (cache / "lock").exists()
    data = mrcfile.open(p, permissive=True).data
    lvl1 = zarr.open_array(LocalStore(str(cache)), path="1", mode="r")
    assert lvl1.shape == (6, 10, 18) and np.array_equal(lvl1[:], block_mean(data, (2, 2, 2)))
    lvl2 = zarr.open_array(LocalStore(str(cache)), path="2", mode="r")
    assert np.array_equal(lvl2[:], block_mean(block_mean(data, (2, 2, 2)), (2, 2, 2)))
    info = json.loads((cache / "info.json").read_text())
    ds = info["attributes"]["ome"]["multiscales"][0]["datasets"]
    assert [d["path"] for d in ds] == ["0", "1", "2"]
    assert ds[1]["coordinateTransformations"] == [{"type": "scale", "scale": [2.0, 2.0, 2.0]},
                                                  {"type": "translation", "translation": [0.5, 0.5, 0.5]}]
    assert info["attributes"]["zarrbridge"]["levels"] == "multiscale"
    bj = json.loads((cache / "build.json").read_text())
    assert bj["state"] == "done" and bj["bytes_read"] > 0
    assert build(p, cfg_min4) == SKIPPED_VALID
    assert build(p, cfg_min4, force=True) == BUILT
    fp = json.loads((cache / "fingerprint.json").read_text())
    assert fp["plan"]["factor"] == [2, 2, 2] and fp["plan"]["level_chunks"][1] == [6, 10, 18]
    op = open_dataset_reader(p, cfg_min4)
    assert validate(cache, op) == ("valid", None)


def test_build_tiff_stack_221(tmp_path, cfg_min4):
    a = stack_u16()
    p = make_tiff(tmp_path / "s.tif", a, compression="lzw", rowsperstrip=32)
    assert build(p, cfg_min4, jobs=2) == BUILT
    cache = tmp_path / ".zarrbridge" / "s.tif"
    lvl1 = zarr.open_array(LocalStore(str(cache)), path="1", mode="r")
    assert lvl1.shape == (5, 50, 60) and lvl1.chunks == (1, 50, 60)
    assert np.array_equal(lvl1[:], block_mean(a, (1, 2, 2)))
    fp = json.loads((cache / "fingerprint.json").read_text())
    assert fp["reader"]["name"] == "tiff" and fp["plan"]["factor"] == [1, 2, 2]


def test_build_refkv_ome_saves_index(tmp_path, cfg_min4):
    a = stack_u16()
    p = make_tiff(tmp_path / "o.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, compression="lzw", rowsperstrip=32)
    assert build(p, cfg_min4) == BUILT
    cache = tmp_path / ".zarrbridge" / "o.ome.tif"
    assert (cache / "index.vzip").exists() and (cache / "index.json").exists()
    lvl1 = zarr.open_array(LocalStore(str(cache)), path="1", mode="r")
    assert np.array_equal(lvl1[:], block_mean(a, (1, 2, 2)))
    op = open_dataset_reader(p, cfg_min4, cache_dir=cache)
    assert op.reader_reason.startswith("refkv: index")


def test_build_saves_index_before_levels(tmp_path, cfg_min4, monkeypatch):
    """The index lands first, so a failed build or a valid pyramid without one still leaves it on disk."""
    from zarrbridge.build import builder as b
    a = stack_u16()
    p = make_tiff(tmp_path / "o.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, compression="lzw", rowsperstrip=32)
    cache = tmp_path / ".zarrbridge" / "o.ome.tif"
    monkeypatch.setattr(b, "_build_locked", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert build(p, cfg_min4) == ERROR
    assert (cache / "index.vzip").exists() and (cache / "index.json").exists()
    monkeypatch.undo()
    assert build(p, cfg_min4) == BUILT
    (cache / "index.vzip").unlink(); (cache / "index.json").unlink()
    assert build(p, cfg_min4) == SKIPPED_VALID  # valid pyramid, missing index: the index is written anyway
    assert (cache / "index.vzip").exists()


def test_build_locked_and_unservable(tmp_path, cfg_min4):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    lock = BuildLock.acquire(tmp_path / ".zarrbridge" / "a.mrc")
    assert build(p, cfg_min4) == SKIPPED_LOCKED
    lock.release()
    (tmp_path / "x.bin").write_bytes(b"\0" * 2000)
    rep = tmp_path / "r.json"
    assert build(tmp_path / "x.bin", cfg_min4, report=rep) == UNSERVABLE
    assert json.loads(rep.read_text())["status"] == "unservable"


@pytest.mark.slow
def test_kill_mid_build_leaves_missing(tmp_path, cfg_min4):
    p = make_mrc(tmp_path / "big.mrc", (256, 256, 64), fill=canonical)
    cache = tmp_path / ".zarrbridge" / "big.mrc"
    proc = subprocess.Popen([sys.executable, "-m", "zarrbridge", "build", str(p), "--min-axis", "4"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline and not list(cache.glob("*.partial")):
        time.sleep(0.01)
        if proc.poll() is not None:
            break
    if proc.poll() is None:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        op = open_dataset_reader(p, cfg_min4)
        assert validate(cache, op)[0] == "missing" and not (cache / "fingerprint.json").exists()
        assert build(p, cfg_min4) == BUILT  # recovers: stale partials removed
    else:
        pytest.skip("build finished before it could be killed")
