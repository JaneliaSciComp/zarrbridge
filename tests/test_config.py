from zarrbridge.server.config import Config, parse_bytes


def test_defaults():
    c = Config.load(file=None, env={})
    assert c.server.port == 8800 and c.cache.mode == "beside" and c.build.auto is True
    assert c.build.auto_max_bytes == 20 * 2**30 and c.limits.chunk_cache_bytes == 256 * 2**20
    assert c.readers.mrc.chunk_shape == (64, 64, 64) and c.readers.tiff.prefer == "refkv"


def test_precedence(tmp_path):
    f = tmp_path / "zb.toml"
    f.write_text('[server]\nport = 1\n[roots]\ndata = "/d"\n[build]\nauto_max_bytes = "1GiB"\n')
    c = Config.load(file=f, env={"ZB_SERVER_PORT": "2"}, overrides={"server": {"port": 3}})
    assert c.server.port == 3 and c.roots == {"data": "/d"} and c.build.auto_max_bytes == 2**30
    assert Config.load(file=f, env={"ZB_SERVER_PORT": "2"}).server.port == 2
    c = Config.load(env={"ZB_ROOTS": "a=/x,b=/y", "ZB_READERS_MRC_CHUNK_SHAPE": "32,32,32", "ZB_BUILD_AUTO": "off"})
    assert c.roots == {"a": "/x", "b": "/y"} and c.readers.mrc.chunk_shape == (32, 32, 32) and c.build.auto is False


def test_parse_bytes():
    assert parse_bytes("20GiB") == 20 * 2**30 and parse_bytes("3MB") == 3_000_000 and parse_bytes(7) == 7
