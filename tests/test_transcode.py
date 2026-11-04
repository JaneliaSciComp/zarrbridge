import json

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype

from tests.conftest import get as get_sync, make_tiff, stack_u16
from zarrbridge.readers.refkv_reader import RefkvReader
from zarrbridge.store.transcode import TranscodeStore

proto = default_buffer_prototype()


async def test_transcode_lzw_ome_tiff(tmp_path, cfg):
    a = stack_u16()
    op = RefkvReader().open(make_tiff(tmp_path / "a.ome.tif", a, ome=True, metadata={"axes": "ZYX"},
                                      compression="lzw", rowsperstrip=32), cfg)
    ts = TranscodeStore(op.store)
    meta = json.loads((await ts.get("0/zarr.json", proto)).to_bytes())
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "zstd"]
    import zstandard
    chunk = zstandard.decompress((await ts.get("0/c/4/3/0", proto)).to_bytes())
    assert len(chunk) == 32 * 120 * 2 and ts.is_transcoded("0/c/0/0/0") and ts.transcoded_arrays == {"0"}
    assert (await ts.get("0/c/9/0/0", proto)) is None

    raw = RefkvReader().open(make_tiff(tmp_path / "r.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, tile=(32, 32)), cfg)
    ts2 = TranscodeStore(raw.store)
    assert (await ts2.get("0/c/0/0/0", proto)).to_bytes() == (await raw.store.get("0/c/0/0/0", proto)).to_bytes()
    assert not ts2.is_transcoded("0/c/0/0/0")


def test_transcode_zarr_roundtrip(tmp_path, cfg):
    a = stack_u16()
    op = RefkvReader().open(make_tiff(tmp_path / "a.ome.tif", a, ome=True, metadata={"axes": "ZYX"},
                                      compression="lzw", rowsperstrip=32), cfg)
    assert np.array_equal(zarr.open_group(TranscodeStore(op.store), mode="r")["0"][:], a)


def test_float64_served_as_float32(tmp_path, cfg_min4):
    cfg = cfg_min4
    from zarrbridge.dataset import Dataset
    a = (np.random.default_rng(0).random((4, 40, 50)) * 1000).astype("<f8")
    p = make_tiff(tmp_path / "f64.tif", a, compression="zlib", rowsperstrip=16)
    ds = Dataset.open(p, cfg)
    meta = json.loads(get_sync(ds.store, "0/zarr.json"))
    assert meta["data_type"] == "float32" and [c["name"] for c in meta["codecs"]] == ["bytes", "zstd"]
    got = zarr.open_group(ds.store, mode="r")["0"][:]
    assert got.dtype == np.float32 and np.array_equal(got, a.astype(np.float32))
    assert ds.source_of("0/c/0/0/0") == "transcoded"
    assert ds.ensure_pyramid("now") == 0 and zarr.open_group(ds.store, mode="r")["1"].dtype == np.float32


def test_numcodecs_zlib_is_transcoded():
    # Neuroglancer's zarr3 reader knows gzip/zlib but not numcodecs.zlib (refkv emits it for TIFF deflate).
    from zarrbridge.store.codecs import needs_transcode
    meta = {"node_type": "array", "data_type": "uint8",
            "codecs": [{"name": "bytes"}, {"name": "numcodecs.zlib", "configuration": {"level": 6}}]}
    assert needs_transcode(meta)
