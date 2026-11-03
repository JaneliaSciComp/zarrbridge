"""Request-path invariants (spec §3.1, §14): one chunk GET reads one chunk's worth of bytes and never builds."""
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import canonical, get, make_mrc, make_tiff, stack_u16
from zarrbridge.dataset import Dataset
from zarrbridge.readers.mrc import read as mrc_read
from zarrbridge.server.config import Config


def _counter_script(path, key):
    return f"""
import os, sys
os.environ['ZB_INSTRUMENT_READS'] = '1'
from zarrbridge.readers.mrc import read as r
from zarrbridge.dataset import Dataset
from zarrbridge.server.config import Config
from tests.conftest import get
ds = Dataset.open({str(path)!r}, Config.load(env={{}}, overrides={{'build': {{'auto': False}}}}))
r.READ_COUNTER[:] = [0, 0]
data = get(ds.store, {key!r})
print(len(data), r.READ_COUNTER[0], r.READ_COUNTER[1])
"""


def _run(script):
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    return [int(x) for x in out.stdout.split()]


def test_request_path_invariant_mrc(tmp_path):
    nx, ny, nz = 300, 100, 70
    p = make_mrc(tmp_path / "a.mrc", (nx, ny, nz), fill=canonical)
    nbytes, calls, read = _run(_counter_script(p, "0/c/0/0/1"))
    assert nbytes == 64**3 * 2
    assert calls <= 64  # one pread per plane (span-wise) at most
    assert read <= 2 * 64**3 * 2 + 64 * (63 * nx + 64) * 2  # <= 2 x chunk + the span over-read


def test_request_path_invariant_tiff(tmp_path):
    p = make_tiff(tmp_path / "s.tif", stack_u16(), compression="lzw", rowsperstrip=32)
    import tifffile
    seg_bytes = tifffile.TiffFile(p).pages[2].databytecounts[1]
    nbytes, calls, read = _run(_counter_script(p, "0/c/2/1/0"))
    assert calls == 1 and read == seg_bytes  # exactly one pread of exactly the segment


def test_missing_cached_level_spawns_nothing(tmp_path, monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned a process")))
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    ds = Dataset.open(p, Config.load(env={}, overrides={"build": {"auto": False}}))
    assert get(ds.store, "1/c/0/0/0") is None and get(ds.store, "3/zarr.json") is None
    assert ds.status()["levels"]["state"] == "native"
