import numpy as np
import pytest
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype

from zarrbridge.store.chunkstore import ChunkStore
from zarrbridge.store.codecs import OUTPUT_CODECS, decode_chunk, encode_chunk, pad_chunk
from zarrbridge.store.keys import array_path_of, level_of, split_chunk_key

proto = default_buffer_prototype()


def test_split_chunk_key():
    assert split_chunk_key("0/c/1/2/3") == ("0", (1, 2, 3))
    assert split_chunk_key("img/0/c/4") == ("img/0", (4,))
    assert split_chunk_key("zarr.json") is None and split_chunk_key("c/1") is None
    assert level_of("img/2") == ("img", 2) and level_of("0") == ("", 0) and level_of("img") is None
    assert array_path_of("img/0/zarr.json") == "img/0" and array_path_of("zarr.json") == ""


async def test_chunkstore_get_and_range():
    st = ChunkStore({"zarr.json": b"{}"}, lambda p, c: bytes(range(10)) if c == (0,) else None, {"0": (2,)})
    assert (await st.get("zarr.json", proto)).to_bytes() == b"{}"
    assert (await st.get("0/c/0", proto)).to_bytes() == bytes(range(10))
    assert (await st.get("0/c/0", proto, RangeByteRequest(2, 5))).to_bytes() == bytes([2, 3, 4])
    assert await st.get("0/c/1", proto) is None and await st.get("0/c/7", proto) is None
    assert sorted([k async for k in st.list()]) == ["zarr.json"]
    assert sorted([k async for k in st.list_dir("")]) == ["zarr.json"]


def test_encode_decode_roundtrip():
    a = np.arange(24, dtype=">u2").reshape(2, 3, 4)
    meta = {"zarr_format": 3, "node_type": "array", "shape": [2, 3, 4], "data_type": "uint16",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 3, 4]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": OUTPUT_CODECS, "attributes": {}}
    assert np.array_equal(decode_chunk(encode_chunk(a), meta, (2, 3, 4)), a)
    gz = [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "gzip", "configuration": {"level": 1}}]
    enc = encode_chunk(a, gz)
    assert np.array_equal(decode_chunk(enc, dict(meta, codecs=gz)), a)
    assert pad_chunk(np.ones((1, 2), "u1"), (2, 3)).tolist() == [[1, 1, 0], [0, 0, 0]]
