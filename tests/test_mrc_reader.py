import hashlib
import json
import os

import mrcfile
import numpy as np
import zarr

from tests.conftest import canonical, get, make_mrc
from zarrbridge.readers.mrc.header import header_hash, parse_header
from zarrbridge.readers.mrc.reader import MrcReader
from zarrbridge.store.codecs import pad_chunk


def test_header_parse_and_hash(tmp_path):
    p = make_mrc(tmp_path / "a.mrc", (80, 60, 40), mode=1, voxel_size_angstrom=6.8, dmin=0, dmax=29999)
    fd = os.open(p, os.O_RDONLY)
    st = os.stat(p)
    h = parse_header(fd, st.st_size, st.st_mtime_ns)
    assert h.shape_zyx == (40, 60, 80) and h.dtype == np.dtype("<i2")
    assert all(abs(v - 6.8) < 1e-5 for v in h.voxel_size_angstrom) and h.dmax == 29999.0
    assert header_hash(fd, h.data_offset) == "sha256:" + hashlib.sha256(open(p, "rb").read(1024)).hexdigest()
    os.close(fd)


def test_cube_chunk_matches_mrcfile_padded(tmp_path, cfg):
    p = make_mrc(tmp_path / "a.mrc", (80, 60, 40), fill=canonical, dmin=0, dmax=29999)
    op = MrcReader().open(p, cfg)
    data = mrcfile.open(p, permissive=True).data
    got = np.frombuffer(get(op.store, "0/c/0/0/1"), "<i2").reshape(64, 64, 64)
    assert np.array_equal(got, pad_chunk(data[0:40, 0:60, 64:80], (64, 64, 64)))
    assert get(op.store, "0/c/0/0/2") is None and get(op.store, "1/c/0/0/0") is None
    meta = json.loads(get(op.store, "zarr.json"))
    ms = meta["attributes"]["ome"]["multiscales"][0]
    assert [a["unit"] for a in ms["axes"]] == ["angstrom"] * 3
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
    assert meta["attributes"]["mrc"]["nx"] == 80
    assert meta["attributes"]["ome"]["omero"]["channels"][0]["window"]["max"] == 29999
    arr = json.loads(get(op.store, "0/zarr.json"))
    assert arr["shape"] == [40, 60, 80] and arr["chunk_grid"]["configuration"]["chunk_shape"] == [64, 64, 64]
    assert op.images[0].native_chunks == (64, 64, 64) and op.classification.state == "servable"
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], data)
    assert np.array_equal(op.read_region("", (slice(0, 40), slice(10, 20), slice(0, 80))), data[:, 10:20, :])
    op.close()


def test_stack_glob_sets_z_voxel(tmp_path, cfg):
    p = make_mrc(tmp_path / "tilt.st", (32, 32, 8), voxel_size_angstrom=2.0, grid_size=(32, 32, 1))
    op = MrcReader().open(p, cfg)
    assert op.images[0].is_stack and op.images[0].scale == [10.0, 2.0, 2.0]
    assert MrcReader.sniff(p, open(p, "rb").read(1024)) and not MrcReader.sniff(tmp_path / "x.tif", b"II*\0" + b"\0" * 300)
