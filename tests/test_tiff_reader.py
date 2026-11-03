import json

import numpy as np
import pytest
import tifffile
import zarr
import zstandard

from tests.conftest import get, make_tiff, stack_u16
from zarrbridge.readers.tiff.reader import TiffReader


def test_lzw_strips_transcoded_to_zstd(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "a.tif", a, compression="lzw", rowsperstrip=32)
    op = TiffReader().open(p, cfg)
    meta = json.loads(get(op.store, "0/zarr.json"))
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 32, 120]
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "zstd"] and "0" in op.transcoded
    assert op.classification.state == "servable_2d"
    last = zstandard.decompress(get(op.store, "0/c/4/3/0"))
    assert len(last) == 32 * 120 * 2 and np.frombuffer(last, "<u2").reshape(32, 120)[:4].tolist() == a[4, 96:100].tolist()
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)
    assert op.images[0].axis_names == ["z", "y", "x"] and op.identity.extras["pages"] == 5
    op.close()


def test_uncompressed_bigendian_passthrough(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "be.tif", a, byteorder=">", rowsperstrip=32)
    op = TiffReader().open(p, cfg)
    meta = json.loads(get(op.store, "0/zarr.json"))
    assert meta["codecs"] == [{"name": "bytes", "configuration": {"endian": "big"}}] and not op.transcoded
    assert get(op.store, "0/c/0/0/0") == a[0, :32].astype(">u2").tobytes()
    assert len(get(op.store, "0/c/0/3/0")) == 32 * 120 * 2
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)


def test_tiled_zlib(tmp_path, cfg):
    a = stack_u16()
    p = make_tiff(tmp_path / "t.tif", a, tile=(32, 32), compression="zlib")
    op = TiffReader().open(p, cfg)
    meta = json.loads(get(op.store, "0/zarr.json"))
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 32, 32]
    assert len(zstandard.decompress(get(op.store, "0/c/2/3/3"))) == 32 * 32 * 2
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)


def test_imagej_hyperstack_axes(tmp_path, cfg):
    a = stack_u16().reshape(1, 5, 1, 100, 120).repeat(2, axis=2)
    p = make_tiff(tmp_path / "ij.tif", a, imagej=True, resolution=(2.0, 2.0),
                  metadata={"axes": "TZCYX", "spacing": 2.0, "unit": "um"})
    op = TiffReader().open(p, cfg)
    img = op.images[0]
    assert img.axis_names == ["c", "z", "y", "x"] and img.shape == (2, 5, 100, 120) and img.native_chunks == (1, 1, 100, 120)
    assert img.scale == [1.0, 2.0, 0.5, 0.5]
    meta = json.loads(get(op.store, "zarr.json"))["attributes"]["ome"]["multiscales"][0]
    assert meta["axes"][1] == {"name": "z", "type": "space", "unit": "micrometer"} and meta["axes"][0] == {"name": "c", "type": "channel"}
    arr = json.loads(get(op.store, "0/zarr.json"))
    assert arr["codecs"][0] == {"name": "transpose", "configuration": {"order": [1, 0, 2, 3]}}
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], np.moveaxis(a[0], 1, 0))


def test_ome_pyramid_and_series(tmp_path, cfg):
    a = stack_u16()
    p = tmp_path / "p.ome.tif"
    with tifffile.TiffWriter(p, ome=True) as tw:
        tw.write(a, subifds=1, metadata={"axes": "ZYX", "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm",
                                         "PhysicalSizeY": 0.5, "PhysicalSizeYUnit": "µm", "PhysicalSizeZ": 2.0, "PhysicalSizeZUnit": "µm"},
                 tile=(32, 32))
        tw.write(a[:, ::2, ::2], subfiletype=1, tile=(32, 32))
        tw.write(a[:2, :50, :60], metadata={"axes": "ZYX"})
    op = TiffReader().open(p, cfg)
    assert [i.path for i in op.images] == ["0", "1"] and op.images[0].levels == ["0", "1"]
    root = json.loads(get(op.store, "zarr.json"))
    assert root["attributes"]["zarrbridge"]["images"] == ["0", "1"]
    ds = json.loads(get(op.store, "0/zarr.json"))["attributes"]["ome"]["multiscales"][0]["datasets"]
    assert ds[0]["coordinateTransformations"] == [{"type": "scale", "scale": [2.0, 0.5, 0.5]}]
    assert ds[1]["coordinateTransformations"] == [{"type": "scale", "scale": [2.0, 1.0, 1.0]},
                                                  {"type": "translation", "translation": [0.0, 0.25, 0.25]}]
    g = zarr.open_group(op.store, mode="r")
    assert np.array_equal(g["0/1"][:], a[:, ::2, ::2]) and np.array_equal(g["1/0"][:], a[:2, :50, :60])


def test_rgb_contig_and_planar(tmp_path, cfg):
    rgb = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    op = TiffReader().open(make_tiff(tmp_path / "rgb.tif", rgb, photometric="rgb", rowsperstrip=32, compression="lzw"), cfg)
    assert op.images[0].axis_names == ["c", "y", "x"] and op.images[0].native_chunks == (3, 32, 120)
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], np.moveaxis(rgb, -1, 0))
    op2 = TiffReader().open(make_tiff(tmp_path / "rgbp.tif", np.moveaxis(rgb, -1, 0), photometric="rgb",
                                      planarconfig="separate", rowsperstrip=32), cfg)
    assert op2.images[0].axis_names == ["c", "y", "x"] and op2.images[0].native_chunks == (1, 32, 120)
    assert np.array_equal(zarr.open_group(op2.store, mode="r")["0"][:], np.moveaxis(rgb, -1, 0))


def test_multifile_ome_tiff_reads_companion_pages(tmp_path, cfg):
    a = stack_u16((2, 3, 100, 120))  # c, z, y, x
    with tifffile.TiffWriter(tmp_path / "multi_C0.ome.tif", ome=True) as tw:
        tw.write(a[0], metadata={"axes": "ZYX"})
    with tifffile.TiffWriter(tmp_path / "multi_C1.ome.tif", ome=True) as tw:
        tw.write(a[1], metadata={"axes": "ZYX"})
    # stitch an OME-XML that declares both files, as Bio-Formats companion sets do
    import uuid
    from xml.sax.saxutils import escape
    ids = {i: str(uuid.uuid4()) for i in range(2)}
    def xml(this):
        planes = "".join(f'<TiffData FirstC="{c}" FirstZ="{z}" IFD="{z}" PlaneCount="1"><UUID FileName="multi_C{c}.ome.tif">urn:uuid:{ids[c]}</UUID></TiffData>'
                         for c in range(2) for z in range(3))
        return (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="urn:uuid:{ids[this]}">'
                f'<Image ID="Image:0" Name="multi"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="120" SizeY="100" SizeZ="3" SizeC="2" SizeT="1">'
                f'<Channel ID="Channel:0:0" SamplesPerPixel="1"/><Channel ID="Channel:0:1" SamplesPerPixel="1"/>{planes}</Pixels></Image></OME>')
    for i in range(2):
        tifffile.tiffcomment(tmp_path / f"multi_C{i}.ome.tif", xml(i))
    op = TiffReader().open(tmp_path / "multi_C0.ome.tif", cfg)
    assert op.images[0].shape == (2, 3, 100, 120)
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)
