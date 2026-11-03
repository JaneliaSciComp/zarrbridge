import os
from pathlib import Path

import numpy as np
import pytest
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
    idx = tmp_path / "cache" / "index.vzip"
    save_index(op, idx)
    op2 = RefkvReader().open(p, cfg, index_path=idx)
    assert op2.reader_reason.startswith("index") and np.array_equal(zarr.open_group(op2.store, mode="r")["0"][:], a)
    os.utime(p, ns=(1, 1))  # identity changed -> index ignored, re-zarrified
    assert RefkvReader().open(p, cfg, index_path=idx).reader_reason == "zarrified"


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
