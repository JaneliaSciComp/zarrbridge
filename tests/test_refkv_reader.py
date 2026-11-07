import os
from pathlib import Path

import numpy as np
import pytest
import refkv
import zarr

from tests.conftest import make_mrc, make_tiff, stack_u16
from zarrbridge.readers.base import ReaderRejected
from zarrbridge.readers.refkv_reader import RefkvReader, save_index
from zarrbridge.readers.registry import open_dataset_reader, sniffable

CZI = Path.home() / "dev/refkv/fixtures/czi/czi_multiscene.czi"


def test_refkv_ome_tiff_root_image(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "a.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    op = RefkvReader().open(p, cfg)
    assert [i.path for i in op.images] == [""] and op.images[0].native_chunks == (1, 32, 32) and op.images[0].levels == ["0"]
    assert op.identity.extras["pins"][0]["size"] == p.stat().st_size and op.identity.extras["format"] == "tiff"
    assert not op.transcoded and op.classification.state == "servable_2d"
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)
    op.close()


def test_refkv_rejects_plain_stack(tmp_path, cfg):
    with pytest.raises(ReaderRejected, match="without OME-XML"):
        RefkvReader().open(make_tiff(tmp_path / "s.tif", stack_u16()), cfg)


def test_index_roundtrip(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "a.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, compression="lzw", rowsperstrip=32)
    op = RefkvReader().open(p, cfg)
    assert "0" in op.transcoded
    cache = tmp_path / "cache"
    assert save_index(op, cache) == cache / "index.vzip"
    op2 = RefkvReader().open(p, cfg, cache_dir=cache)
    assert op2.reader_reason == "index index.vzip" and np.array_equal(zarr.open_group(op2.store, mode="r")["0"][:], a)
    with pytest.raises(ValueError):
        save_index(op2, cache)  # reopened from an index: nothing to save
    os.utime(p, ns=(1, 1))  # identity changed -> index ignored, re-zarrified
    assert RefkvReader().open(p, cfg, cache_dir=cache).reader_reason == "zarrified"


@pytest.mark.skipif("phkv" not in refkv.BINDINGS, reason="this refkv has no phkv binding")
def test_index_phkv_binding(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "a.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, compression="lzw", rowsperstrip=32)
    cache = tmp_path / "cache"
    save_index(RefkvReader().open(p, cfg), cache, "vzip")
    assert save_index(RefkvReader().open(p, cfg), cache, "phkv") == cache / "index.phkv"
    assert sorted(f.name for f in cache.iterdir()) == ["index.json", "index.phkv"]  # the vzip is replaced
    op = RefkvReader().open(p, cfg, cache_dir=cache)
    assert op.reader_reason == "index index.phkv" and np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)


def test_index_unknown_binding(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "a.ome.tif", a, ome=True, metadata={"axes": "ZYX"}, compression="lzw", rowsperstrip=32)
    with pytest.raises(ValueError, match="cannot write"):
        save_index(RefkvReader().open(p, cfg), tmp_path / "cache", "nope")


def test_select_falls_back_to_tifffile(tmp_path, cfg):
    op = open_dataset_reader(make_tiff(tmp_path / "s.tif", stack_u16()), cfg)
    assert op.reader_name == "tiff" and "refkv rejected" in op.reader_reason


def test_select_mrc_and_no_reader(tmp_path, cfg):
    assert open_dataset_reader(make_mrc(tmp_path / "a.mrc", (8, 8, 8)), cfg).reader_name == "mrc"
    (tmp_path / "x.bin").write_bytes(b"\0" * 100)
    with pytest.raises(ReaderRejected, match="no reader"):
        open_dataset_reader(tmp_path / "x.bin", cfg)
    assert sniffable(tmp_path / "a.mrc") and not sniffable(tmp_path / "x.bin")


@pytest.mark.skipif(not CZI.exists(), reason="refkv fixtures not checked out")
def test_czi_multiscene_collection(cfg):
    op = open_dataset_reader(CZI, cfg)
    assert op.reader_name == "refkv" and [i.path for i in op.images] == ["0", "1", "2", "3", "4"]
    g = zarr.open_group(op.store, mode="r")
    assert g["0/0"].shape == tuple(op.images[0].shape)


def test_bare_arrays_become_images(tmp_path, cfg):
    np.save(tmp_path / "a.npy", np.arange(20 * 30, dtype="<i2").reshape(20, 30))
    op = open_dataset_reader(tmp_path / "a.npy", cfg)
    assert op.reader_name == "refkv" and [i.path for i in op.images] == ["array"] and op.images[0].levels == ["0"]
    g = zarr.open_group(op.store, mode="r")
    assert g["array"].attrs["ome"]["multiscales"][0]["axes"][0]["name"] == "y"
    assert np.array_equal(g["array/0"][:], np.arange(600, dtype="<i2").reshape(20, 30))
    from ome_zarr_models.v05.image import Image
    Image.from_zarr(g["array"])
