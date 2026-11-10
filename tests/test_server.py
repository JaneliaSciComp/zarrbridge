import json
import os
import threading
import time

import mrcfile
import numpy as np
import pytest
import zarr
from starlette.testclient import TestClient

from tests.conftest import canonical, make_mrc, make_tiff, stack_u16
from zarrbridge.build.builder import BUILT, build
from zarrbridge.server.app import create_app
from zarrbridge.server.config import Config


def cfg_root(name, path, **over):
    o = {"roots": {name: str(path)}, "build": {"auto": False, "min_axis": 4, "max_levels": 3}, "server": {}}
    for k, v in over.items():
        sect, _, fld = k.partition("__")
        o.setdefault(sect, {})[fld] = v
    return Config.load(env={}, overrides=o)


def test_serve_mrc_end_to_end(tmp_path):
    p = make_mrc(tmp_path / "data" / "a.mrc", (80, 60, 40), fill=canonical)
    c = TestClient(create_app(cfg_root("data", tmp_path / "data")))
    r = c.get("/data/a.mrc/zarr.json")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache" and r.headers["etag"].startswith('W/"')
    assert r.headers["content-type"].startswith("application/json")
    assert c.get("/data/a.mrc/zarr.json", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    r = c.get("/data/a.mrc/0/c/0/0/0")
    assert r.status_code == 200 and len(r.content) == 64**3 * 2
    assert r.headers["x-zarrbridge-source"] == "native" and r.headers["x-zarrbridge-read"] == "spanwise"
    assert r.headers["cache-control"] == "public, max-age=3600, must-revalidate"
    assert c.get("/data/a.mrc/0/c/9/9/9").status_code == 404 and c.get("/data/a.mrc/1/c/0/0/0").status_code == 404
    assert c.head("/data/a.mrc/0/c/0/0/0").headers["content-length"] == str(64**3 * 2)
    r = c.get("/data/a.mrc/0/c/0/0/0", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and len(r.content) == 10 and r.headers["content-range"] == "bytes 0-9/*"
    assert c.get("/data/a.mrc/0/c/0/0/0", headers={"Range": "bytes=-4"}).content == c.get("/data/a.mrc/0/c/0/0/0").content[-4:]
    assert c.get("/data/../etc/passwd").status_code == 404 and c.get("/data/a.mrc/%2e%2e/a.mrc/zarr.json").status_code == 404 and c.get("/data/%2e%2e/etc/passwd").status_code == 404
    assert c.get("/nope/a.mrc/zarr.json").status_code == 404
    s = c.get("/_status/data/a.mrc").json()
    assert s["reader"] == "mrc" and s["levels"]["state"] == "native"
    assert s["pyramid"]["state"] == "not_built" and s["index"]["state"] == "none"
    h = c.get("/_health").json()
    assert h["ok"] is True and h["datasets_open"] == 1
    assert build(p, cfg_root("data", tmp_path / "data")) == BUILT
    time.sleep(0.01)
    r = c.get("/data/a.mrc/zarr.json", headers={"If-None-Match": r.headers.get("etag", "x")})
    meta = c.get("/data/a.mrc/zarr.json").json()
    assert len(meta["attributes"]["ome"]["multiscales"][0]["datasets"]) >= 2  # recheck(force) is not needed: validity re-read
    r = c.get("/data/a.mrc/1/c/0/0/0")
    assert r.status_code == 200 and r.headers["x-zarrbridge-source"] == "cache" and r.headers["cache-control"] == "public, max-age=86400, must-revalidate"
    s = c.get("/_status/data/a.mrc").json()
    assert s["levels"]["state"] == "multiscale" and s["pyramid"]["state"] == "built"


def test_tiff_transcoded_header_and_unservable(tmp_path):
    make_tiff(tmp_path / "d" / "s.tif", stack_u16(), compression="lzw", rowsperstrip=32)
    (tmp_path / "d" / "junk.bin").write_bytes(b"\0" * 4000)
    c = TestClient(create_app(cfg_root("d", tmp_path / "d")))
    r = c.get("/d/s.tif/0/c/0/0/0")
    assert r.status_code == 200 and r.headers["x-zarrbridge-source"] == "transcoded" and r.headers["x-zarrbridge-read"] == "segment"
    assert c.get("/d/s.tif/zarr.json").json()["attributes"]["zarrbridge"]["levels"] == "native"
    assert c.get("/_status/d/s.tif").json()["advice"] == ["xy_only"]
    assert c.get("/d/junk.bin/zarr.json").status_code == 422
    assert c.get("/_status/d/junk.bin").json()["levels"]["state"] == "unservable"


def test_token_required(tmp_path):
    make_mrc(tmp_path / "d" / "a.mrc", (8, 8, 8))
    c = TestClient(create_app(cfg_root("d", tmp_path / "d", server__token="s3cret")))
    assert c.get("/d/a.mrc/zarr.json").status_code == 401
    assert c.get("/d/a.mrc/zarr.json?token=s3cret").status_code == 200
    assert c.get("/d/a.mrc/zarr.json", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    with pytest.raises(ValueError, match="insecure"):
        create_app(cfg_root("d", tmp_path / "d", server__host="0.0.0.0"))


def test_symlink_escape_404(tmp_path):
    outside = make_mrc(tmp_path / "outside" / "o.mrc", (8, 8, 8))
    (tmp_path / "d").mkdir()
    os.symlink(outside, tmp_path / "d" / "link.mrc")
    assert TestClient(create_app(cfg_root("d", tmp_path / "d"))).get("/d/link.mrc/zarr.json").status_code == 404
    assert TestClient(create_app(cfg_root("d", tmp_path / "d", server__follow_symlinks=True))).get("/d/link.mrc/zarr.json").status_code == 200


def test_browse_lists_files(tmp_path):
    make_mrc(tmp_path / "d" / "sub" / "a.mrc", (8, 8, 8))
    c = TestClient(create_app(cfg_root("d", tmp_path / "d", server__cors_origins=["*"])))
    assert "d/" in c.get("/").text
    html = c.get("/d/sub/").text
    assert "a.mrc" in html and ("neuroglancer-demo" in html or "/neuroglancer/#!" in html) and "vizarr" in html
    import re, json as _json
    from urllib.parse import unquote
    state = _json.loads(unquote(re.search(r'(?:neuroglancer-demo\.appspot\.com/|/neuroglancer/)#!([^"]+)"', html).group(1)))
    assert state["layers"][0]["tab"] == "rendering" and state["selectedLayer"] == {"visible": True, "layer": "a.mrc"}
    vd = tmp_path / "viewer"; vd.mkdir(); (vd / "index.html").write_text("<html>ng</html>")
    c2 = TestClient(create_app(cfg_root("d", tmp_path / "d", server__viewer_dir=str(vd))))
    html2 = c2.get("/d/sub/").text
    assert "/neuroglancer/#!" in html2 and c2.get("/neuroglancer/").text == "<html>ng</html>"
    r = c.options("/d/sub/a.mrc/zarr.json", headers={"Origin": "http://x", "Access-Control-Request-Method": "GET",
                                                    "Access-Control-Request-Private-Network": "true"})
    assert r.status_code in (200, 204) and r.headers.get("access-control-allow-private-network") == "true"
    assert c.get("/d/sub/a.mrc", follow_redirects=False).status_code == 302
    assert c.get("/d/missing/").status_code == 404


def test_cors_headers(tmp_path):
    make_mrc(tmp_path / "d" / "a.mrc", (8, 8, 8))
    c = TestClient(create_app(cfg_root("d", tmp_path / "d", server__cors_origins=["*"])))
    r = c.get("/d/a.mrc/zarr.json", headers={"Origin": "http://x"})
    assert r.headers["access-control-allow-origin"] == "*" and "ETag" in r.headers["access-control-expose-headers"]


def test_zarr_python_over_http(tmp_path):
    import uvicorn
    p = make_mrc(tmp_path / "d" / "a.mrc", (80, 60, 40), fill=canonical)
    app = create_app(cfg_root("d", tmp_path / "d"))
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        g = zarr.open_group(f"http://127.0.0.1:{port}/d/a.mrc", mode="r")
        assert np.array_equal(g["0"][:], mrcfile.open(p, permissive=True).data)
    finally:
        server.should_exit = True
        t.join(timeout=5)


def test_neuroglancer_state_is_auto_layer():
    from zarrbridge.server.browse import neuroglancer_state
    st = neuroglancer_state("https://h/x.tif", "x.tif")
    assert st["layers"] == [{"type": "auto", "source": "zarr3://https://h/x.tif", "name": "x.tif", "tab": "rendering"}]
    assert st["selectedLayer"] == {"visible": True, "layer": "x.tif"}


def test_viewer_passthrough(tmp_path, monkeypatch):
    from zarrbridge.server import app as appmod
    calls = []

    def fake_fetch(self, path):
        calls.append(path)
        if path == "index.html":
            return 200, b"<title>neuroglancer</title>", {"Content-Type": "text/html"}
        if path == "main.0123456789ab.js":
            import gzip
            return 200, gzip.compress(b"js"), {"content-type": "application/javascript", "content-encoding": "gzip",
                                               "cache-control": "public, max-age=600"}
        return 404, b"", {}
    monkeypatch.setattr(appmod.ViewerProxy, "fetch", fake_fetch)
    make_mrc(tmp_path / "d" / "a.mrc", (8, 8, 8))
    c = TestClient(create_app(cfg_root("d", tmp_path / "d")))
    assert c.get("/neuroglancer", follow_redirects=False).status_code == 307
    assert c.get("/neuroglancer/").text == "<title>neuroglancer</title>"
    r = c.get("/neuroglancer/main.0123456789ab.js")
    assert r.content == b"js"  # httpx inflated it: Content-Encoding was forwarded
    assert r.headers.get_list("cache-control") == ["public, max-age=31536000, immutable"]  # replaced, not duplicated
    c.get("/neuroglancer/main.0123456789ab.js")
    assert calls.count("main.0123456789ab.js") == 1  # hashed asset served from cache
    assert c.get("/neuroglancer/missing.js").status_code == 404 and c.get("/neuroglancer/%2e%2e/x").status_code in (400, 404)
    assert "/neuroglancer/#!" in c.get("/d/").text
    off = TestClient(create_app(cfg_root("d", tmp_path / "d", server__viewer_url="")))
    assert off.get("/neuroglancer/").status_code == 404 and "neuroglancer-demo.appspot.com/#!" in off.get("/d/").text


def _slow_zarrify(monkeypatch, delay):
    import zarrbridge.readers.refkv_reader as rr
    real = rr._zarrify
    calls = []

    def z(path, config, timeout):
        calls.append(path)
        time.sleep(delay)
        return real(path, config, timeout)
    monkeypatch.setattr(rr, "_zarrify", z)
    return calls


def test_status_pending_shape(tmp_path, monkeypatch):
    make_tiff(tmp_path / "data" / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    _slow_zarrify(monkeypatch, 1.0)
    c = TestClient(create_app(cfg_root("data", tmp_path / "data", limits__open_grace_s=0.1)))
    t0 = time.time()
    s = c.get("/_status/data/a.ome.tif").json()
    assert time.time() - t0 < 0.8
    assert s["classification"] == "pending" and s["levels"]["state"] == "indexing" and s["levels"]["elapsed_s"] >= 0
    assert "gives up after" in s["levels"]["reason"].lower() and s["cache"]["mode"] == "beside"
    assert s["index"]["state"] == "reading" and ";" not in s["index"]["reason"]
    assert s["pyramid"]["state"] == "not_built"
    assert c.get("/_status/data/a.ome.tif/0").json()["levels"]["state"] == "indexing"
    time.sleep(1.2)
    assert c.get("/_status/data/a.ome.tif").json()["levels"]["state"] == "native"


def test_status_of_fast_file_is_complete_on_first_request(tmp_path):
    make_mrc(tmp_path / "data" / "a.mrc", (16, 16, 16))
    c = TestClient(create_app(cfg_root("data", tmp_path / "data")))
    assert c.get("/_status/data/a.mrc").json()["levels"]["state"] == "native"


def test_open_timeout_is_unindexed(tmp_path, monkeypatch):
    make_tiff(tmp_path / "data" / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    import zarrbridge.readers.refkv_reader as rr
    from zarrbridge.readers.base import IndexTimeout
    monkeypatch.setattr(rr, "_zarrify", lambda path, config, timeout: (_ for _ in ()).throw(IndexTimeout(path, 60)))
    c = TestClient(create_app(cfg_root("data", tmp_path / "data")))
    s = c.get("/_status/data/a.ome.tif").json()
    assert s["classification"] == "unindexed" and s["levels"]["state"] == "unindexed"
    assert "too big to index automatically" in s["levels"]["reason"] and "indexed manually" in s["levels"]["reason"]
    assert s["index"]["state"] == "failed" and s["index"]["reason"] == s["levels"]["reason"]
    assert ";" not in s["index"]["reason"]
    r = c.get("/data/a.ome.tif/zarr.json")
    assert r.status_code == 422 and r.text.startswith("unindexed:")


def test_pending_open_does_not_block_other_requests(tmp_path, monkeypatch):
    make_tiff(tmp_path / "data" / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    make_mrc(tmp_path / "data" / "b.mrc", (16, 16, 16))
    _slow_zarrify(monkeypatch, 1.5)
    c = TestClient(create_app(cfg_root("data", tmp_path / "data", limits__open_grace_s=0.1)))
    done = []
    ts = [threading.Thread(target=lambda: done.append(c.get("/data/a.ome.tif/zarr.json").status_code)) for _ in range(40)]
    [t.start() for t in ts]
    time.sleep(0.3)                       # 40 key requests are now waiting on one open
    t0 = time.time()
    assert c.get("/data/b.mrc/zarr.json").status_code == 200 and c.get("/_health").status_code == 200
    assert c.get("/_status/data/a.ome.tif").json()["levels"]["state"] == "indexing"
    assert time.time() - t0 < 0.5         # served while a.ome.tif was still opening, with no pool to exhaust
    [t.join() for t in ts]
    assert done == [200] * 40
