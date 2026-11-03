"""Every served hierarchy validates as NGFF 0.5 and round-trips through zarr-python."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from ome_zarr_models.v05.image import Image

from tests.conftest import canonical, get, make_mrc, make_tiff, stack_u16
from zarrbridge.build.builder import BUILT, build
from zarrbridge.dataset import Dataset

CZI = Path.home() / "dev/refkv/fixtures/czi/czi_multiscene.czi"


def _fixtures(tmp_path):
    a = stack_u16()
    yield "mrc_cubes", make_mrc(tmp_path / "a.mrc", (36, 20, 12), fill=canonical, dmin=0, dmax=100)
    yield "tiff_lzw_stack", make_tiff(tmp_path / "s.tif", a, compression="lzw", rowsperstrip=32)
    yield "tiff_tiled", make_tiff(tmp_path / "t.tif", a, tile=(32, 32), compression="zlib")
    yield "imagej", make_tiff(tmp_path / "ij.tif", a.reshape(1, 5, 1, 100, 120).repeat(2, axis=2), imagej=True,
                              metadata={"axes": "TZCYX", "spacing": 2.0, "unit": "um"})
    yield "refkv_ome", make_tiff(tmp_path / "o.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, compression="lzw", rowsperstrip=32)
    p = tmp_path / "pyr.ome.tif"
    with tifffile.TiffWriter(p, ome=True) as tw:
        tw.write(a, subifds=1, metadata={"axes": "ZYX"}, tile=(32, 32))
        tw.write(a[:, ::2, ::2], subfiletype=1, tile=(32, 32))
        tw.write(a[:2, :50, :60], metadata={"axes": "ZYX"})
    yield "ome_pyramid_series", p
    if CZI.exists():
        yield "czi_multiscene", CZI


def _validate(ds):
    g = zarr.open_group(ds.store, mode="r")
    for img in ds.opened.images:
        node = g if img.path == "" else g[img.path]
        model = Image.from_zarr(node)
        ms = model.ome_attributes.multiscales[0]
        for d in ms.datasets:
            arr = node[d.path]
            meta = json.loads(get(ds.store, f"{img.level_path(d.path)}/zarr.json"))
            assert tuple(arr.shape) == tuple(meta["shape"])
            _ = arr[tuple(slice(0, min(2, s)) for s in arr.shape)]  # chunk read works


@pytest.mark.parametrize("name", ["mrc_cubes", "tiff_lzw_stack", "tiff_tiled", "imagej", "refkv_ome", "ome_pyramid_series", "czi_multiscene"])
def test_ngff_05_conformance_native_and_built(name, tmp_path, cfg_min4):
    paths = dict(_fixtures(tmp_path))
    if name not in paths:
        pytest.skip("fixture unavailable")
    p = paths[name]
    ds = Dataset.open(p, cfg_min4)
    _validate(ds)
    if name != "czi_multiscene":
        assert build(p, cfg_min4) == BUILT
        ds.recheck(force=True)
        assert ds.validity == "valid"
        _validate(ds)
        g = zarr.open_group(ds.store, mode="r")
        img = ds.opened.images[0]
        if len(img.levels) == 1:  # a built level exists and is smaller
            node = g if img.path == "" else g[img.path]
            assert node["1"].shape < node["0"].shape
    ds.close()
