from pathlib import Path

from zarrbridge.build.plan import choose_factor, plan_levels
from zarrbridge.readers.base import ImageInfo
from zarrbridge.server.config import Config
from zarrbridge.store.omezarr import axis

ZYX = [axis("z", "micrometer"), axis("y", "micrometer"), axis("x", "micrometer")]


def img(shape, chunks, scale=(1, 1, 1), is_stack=False):
    return ImageInfo(path="", shape=shape, dtype="uint16", axes=ZYX, scale=list(scale), native_chunks=chunks,
                     codecs=[], levels=["0"], is_stack=is_stack)


def test_plan_auto_2d_chunks_gives_221_tiles():
    cfg = Config.load(env={})
    i = img((1000, 4096, 4096), (1, 128, 4096))
    assert choose_factor(i, cfg, Path("a.tif")) == (1, 2, 2)
    lv = plan_levels(i, (1, 2, 2), min_axis=64, max_levels=8)
    assert lv[1].shape == (1000, 2048, 2048) and lv[1].chunks == (1, 512, 512)
    assert lv[1].scale == [1.0, 2.0, 2.0] and lv[1].translation == [0.0, 0.5, 0.5]
    assert len(lv) == 7 and lv[6].shape == (1000, 64, 64) and lv[6].chunks == (1, 64, 64)


def test_plan_cubes_isotropic_gives_222():
    cfg = Config.load(env={})
    i = img((500, 4000, 4000), (64, 64, 64), scale=(6.8, 6.8, 6.8))
    assert choose_factor(i, cfg, Path("a.mrc")) == (2, 2, 2)
    lv = plan_levels(i, (2, 2, 2), min_axis=64, max_levels=8)
    assert lv[1].shape == (250, 2000, 2000) and lv[1].chunks == (64, 64, 64)
    assert lv[3].shape == (63, 500, 500) and lv[3].chunks == (63, 64, 64)
    assert lv[4].shape == (63, 250, 250)  # z stopped at < 64
    assert lv[2].translation == [6.8 * 1.5] * 3


def test_plan_policy_cases():
    cfg = Config.load(env={})
    assert choose_factor(img((100, 512, 512), (64, 64, 64), scale=(5, 1, 1)), cfg, "a.mrc") == (1, 2, 2)  # anisotropic z
    assert choose_factor(img((100, 512, 512), (64, 64, 64), is_stack=True), cfg, "a.st") == (1, 2, 2)
    cfg2 = Config.load(env={}, overrides={"build": {"factor_by_glob": {"*.st": "2,2,2"}}})
    assert choose_factor(img((100, 512, 512), (1, 512, 512), is_stack=True), cfg2, "/x/tilt.st") == (2, 2, 2)
    cfg3 = Config.load(env={}, overrides={"build": {"factor": "2,2,1"}})
    assert choose_factor(img((100, 512, 512), (64, 64, 64)), cfg3, "a.mrc") == (1, 2, 2)
    tcyx = ImageInfo(path="", shape=(3, 2, 1000, 1000), dtype="uint8", axes=[axis("t"), axis("c"), axis("y"), axis("x")],
                     scale=[1, 1, 1, 1], native_chunks=(1, 1, 64, 1000), codecs=[], levels=["0"])
    assert choose_factor(tcyx, cfg, "a.tif") == (1, 1, 2, 2)
    lv = plan_levels(tcyx, (1, 1, 2, 2), min_axis=64)
    assert lv[1].chunks == (1, 1, 500, 500) and lv[1].scale == [1.0, 1.0, 2.0, 2.0] and lv[1].translation == [0.0, 0.0, 0.5, 0.5]


def test_plan_keeps_level0_translation():
    i = ImageInfo(path="", shape=(3, 4096, 4096), dtype="uint16", axes=[axis("c"), axis("y", "micrometer"), axis("x", "micrometer")],
                  scale=[1.0, 0.5, 0.5], native_chunks=(1, 1024, 1024), codecs=[], levels=["0"], translation=[0.0, 100.0, 200.0])
    lv = plan_levels(i, (1, 2, 2), min_axis=64)
    assert lv[0].translation == [0.0, 100.0, 200.0]
    assert lv[1].translation == [0.0, 100.25, 200.25] and lv[2].translation == [0.0, 100.75, 200.75]
