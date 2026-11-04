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
    assert s["reader"] == "mrc" and s["levels"]["state"] == "native" and "zarrbridge build" in s["levels"]["reason"]
    h = c.get("/_health").json()
    assert h["ok"] is True and h["datasets_open"] == 1
    assert build(p, cfg_root("data", tmp_path / "data")) == BUILT
    time.sleep(0.01)
    r = c.get("/data/a.mrc/zarr.json", headers={"If-None-Match": r.headers.get("etag", "x")})
    meta = c.get("/data/a.mrc/zarr.json").json()
    assert len(meta["attributes"]["ome"]["multiscales"][0]["datasets"]) >= 2  # recheck(force) is not needed: validity re-read
    r = c.get("/data/a.mrc/1/c/0/0/0")
    assert r.status_code == 200 and r.headers["x-zarrbridge-source"] == "cache" and r.headers["cache-control"] == "public, max-age=86400, must-revalidate"
    assert c.get("/_status/data/a.mrc").json()["levels"]["state"] == "multiscale"


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


def test_neuroglancer_state_colours_channels():
    from zarrbridge.server.browse import neuroglancer_state
    meta = {"attributes": {"ome": {"multiscales": [{"axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}]}],
                                   "omero": {"channels": [{"label": "DAPI", "color": "0000FF", "window": {"min": 0, "max": 65535, "start": 100, "end": 4000}},
                                                          {"label": "GFP", "color": "00FF00", "active": False}]}}}}
    st = neuroglancer_state("https://h/x.tif", "x.tif", meta)
    assert [l["name"] for l in st["layers"]] == ["DAPI", "GFP"]
    assert st["layers"][0]["localPosition"] == [0] and st["layers"][0]["localDimensions"] == {"c'": [1, ""]}
    assert st["layers"][0]["shaderControls"] == {"color": "#0000FF", "contrast": {"range": [100, 4000], "window": [0, 65535]}}
    assert st["layers"][1].get("archived") is True and st["layers"][0]["blend"] == "additive"
    rgb = {"attributes": {"ome": {"multiscales": [{"axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}]}]}}, "zarrbridge_shape": {"c": 3}}
    assert [l["shaderControls"]["color"] for l in neuroglancer_state("u", "n", rgb)["layers"]] == ["#ff0000", "#00ff00", "#0000ff"]
    assert len(neuroglancer_state("u", "n", None)["layers"]) == 1
