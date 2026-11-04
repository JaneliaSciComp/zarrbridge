"""Parity with mrc-ng-server: byte-identical level-0 cubes and level>=1 chunks; documented metadata diffs only.
Requires the sibling checkout at ~/dev/mrc-ng-server (skipped otherwise)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zstandard
from zarr.storage import LocalStore

from tests.conftest import canonical, get, make_mrc
from zarrbridge.build.builder import BUILT, build
from zarrbridge.dataset import Dataset
from zarrbridge.server.config import Config

MRCNG = Path.home() / "dev/mrc-ng-server/src"
pytestmark = pytest.mark.skipif(not MRCNG.exists(), reason="mrc-ng-server checkout not found")


@pytest.fixture(scope="module")
def mrcng():
    """mrc-ng-server must be importable (`uv pip install --no-deps -e ~/dev/mrc-ng-server`)."""
    try:
        import mrcng.pyramid, mrcng.fingerprint, mrcng.omezarr, mrcng.reader, mrcng.mrcheader  # noqa
    except Exception as e:  # ImportError or PackageNotFoundError
        pytest.skip(f"mrc-ng-server not importable: {e}")
    return sys.modules["mrcng"]


# mrc-ng-server stops each axis individually at min_axis; zarrbridge halves all axes together (derivation 4), so
# the cases stay within the levels where both plans agree.
@pytest.mark.parametrize("shape,chunk,min_axis,levels", [((32, 32, 32), 8, 8, 3), ((36, 20, 12), 8, 4, 3), ((101, 37, 13), 16, 4, 3)])
def test_parity_level0_and_pyramid(tmp_path, mrcng, shape, chunk, min_axis, levels):
    from mrcng.fingerprint import Params
    from mrcng.pyramid import build_one
    src = tmp_path / "src"
    p = make_mrc(src / "a.mrc", shape, fill=canonical, voxel_size_angstrom=20.0)
    params = Params(chunk_size=(chunk,) * 3, downsample="mean", min_axis_size=min_axis, max_levels=levels, dtype="int16", encoding="raw")
    res = build_one(src, tmp_path / "ngcache", "a.mrc", params, formats=("omezarr",))
    ng_dir = Path(res.cache_dir) if hasattr(res, "cache_dir") else next((tmp_path / "ngcache").glob("*/*"))
    ng_group = json.loads((ng_dir / "omezarr" / "zarr.json").read_text())

    cfg = Config.load(env={}, overrides={"build": {"min_axis": min_axis, "max_levels": levels, "auto": False},
                                         "readers": {"mrc": {"chunk_shape": [chunk] * 3}}})
    assert build(p, cfg) == BUILT
    ds = Dataset.open(p, cfg)
    assert ds.validity == "valid"
    zb_group = json.loads(get(ds.store, "zarr.json"))
    ng_ds = ng_group["attributes"]["ome"]["multiscales"][0]["datasets"]
    zb_ds = zb_group["attributes"]["ome"]["multiscales"][0]["datasets"]
    assert [d["path"] for d in ng_ds] == [d["path"] for d in zb_ds]
    for a, b in zip(ng_ds, zb_ds):  # documented diff: nanometer vs angstrom (x10)
        for ta, tb in zip(a["coordinateTransformations"], b["coordinateTransformations"]):
            assert ta["type"] == tb["type"]
            np.testing.assert_allclose(np.array(ta.get("scale", ta.get("translation"))) * 10, tb.get("scale", tb.get("translation")))
    assert [ax["name"] for ax in ng_group["attributes"]["ome"]["multiscales"][0]["axes"]] == \
           [ax["name"] for ax in zb_group["attributes"]["ome"]["multiscales"][0]["axes"]]

    # level 0: byte-identical with mrc-ng's live path (read_chunk + pad_chunk, raw little-endian)
    import os
    from mrcng.mrcheader import parse_header
    from mrcng.omezarr import pad_chunk as ng_pad
    from mrcng.reader import read_chunk as ng_read
    fd = os.open(p, os.O_RDONLY)
    st = os.stat(p)
    hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
    arr0 = json.loads(get(ds.store, "0/zarr.json"))
    assert arr0["chunk_grid"]["configuration"]["chunk_shape"] == [chunk] * 3 and arr0["data_type"] == "int16"
    nz, ny, nx = hdr.shape_zyx
    for kz in range(-(-nz // chunk)):
        for ky in range(-(-ny // chunk)):
            for kx in range(-(-nx // chunk)):
                ng = ng_pad(ng_read(fd, hdr, kx * chunk, (kx + 1) * chunk, ky * chunk, (ky + 1) * chunk, kz * chunk, (kz + 1) * chunk), (chunk,) * 3).tobytes()
                assert get(ds.store, f"0/c/{kz}/{ky}/{kx}") == ng
    os.close(fd)

    # levels >= 1: our zstd chunks decompress to mrc-ng's raw chunk files. Documented diff: the spec clamps the
    # chunk shape to the level shape (mrc-ng pads to the full chunk), so compare over our (clamped) extent.
    for lvl in range(1, len(zb_ds)):
        ng_level = ng_dir / "omezarr" / str(lvl) / "c"
        files = [f for f in sorted(ng_level.rglob("*")) if f.is_file()]
        assert files, f"mrc-ng wrote no chunks for level {lvl}"
        our_chunk = tuple(json.loads(get(ds.store, f"{lvl}/zarr.json"))["chunk_grid"]["configuration"]["chunk_shape"])
        for f in files:
            key = f"{lvl}/c/" + "/".join(f.relative_to(ng_level).parts)
            ours = get(ds.store, key)
            assert ours is not None, key
            ng = np.frombuffer(f.read_bytes(), "<i2").reshape((chunk,) * 3)[: our_chunk[0], : our_chunk[1], : our_chunk[2]]
            assert zstandard.decompress(ours) == ng.tobytes(), key
            if our_chunk == (chunk,) * 3:
                assert zstandard.decompress(ours) == f.read_bytes(), key  # byte-identical when no clamping applies
    ds.close()
