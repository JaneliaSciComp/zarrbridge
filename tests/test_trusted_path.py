from starlette.testclient import TestClient

from tests.conftest import make_mrc
from zarrbridge.server.app import create_app
from zarrbridge.server.config import Config


def test_trusted_path_mode(tmp_path):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    cfg = Config.load(env={}, overrides={"server": {"trusted_path_header": "X-Fg-Resolved-Path", "browse": False,
                                                      "allow_prefixes": [str(tmp_path) + "/**"]},
                                           "build": {"auto": False}})
    c = TestClient(create_app(cfg))
    assert c.get("/zarr.json").status_code == 400
    H = {"X-Fg-Resolved-Path": str(p)}
    assert c.get("/zarr.json", headers=H).status_code == 200
    assert c.get("/0/c/0/0/0", headers=H).status_code == 200
    assert c.get("/zarr.json", headers={"X-Fg-Resolved-Path": "/etc/passwd"}).status_code == 403
    assert c.get("/zarr.json", headers={"X-Fg-Resolved-Path": str(tmp_path / "missing.mrc")}).status_code == 404
    assert c.get("/_status", headers=H).json()["reader"] == "mrc"
    assert c.get("/_status").status_code == 400
    assert c.get("/_health").json()["mode"] == "trusted-path"
