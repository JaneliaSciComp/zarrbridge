import json

import numpy as np

from tests.conftest import canonical, make_mrc, make_tiff, stack_u16
from zarrbridge.cli import main


def test_cli_build_status_prune(tmp_path, capsys):
    p = make_mrc(tmp_path / "a.mrc", (36, 20, 12), fill=canonical)
    assert main(["build", str(p), "--min-axis", "4"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out.strip().splitlines()[-1])["status"] == "built"
    assert main(["status", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "multiscale" in out and "valid" in out
    with open(p, "ab") as f:
        f.write(b"\0")
    assert main(["status", str(tmp_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["fingerprint"] == "stale" and rows[0]["mismatch"] == "size"
    assert main(["prune", str(tmp_path), "--stale", "--dry-run"]) == 0
    assert (tmp_path / ".zarrbridge" / "a.mrc").exists() and "would remove" in capsys.readouterr().out
    assert main(["prune", str(tmp_path), "--stale"]) == 0
    assert not (tmp_path / ".zarrbridge" / "a.mrc").exists()


def test_cli_build_dir_and_exit_codes(tmp_path, capsys):
    make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    make_tiff(tmp_path / "s.tif", stack_u16(), compression="lzw", rowsperstrip=32)
    (tmp_path / "x.bin").write_bytes(b"\0" * 3000)
    assert main(["build", str(tmp_path), "--min-axis", "4"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert {l["status"] for l in lines} == {"built"} and len(lines) == 2
    assert main(["build", str(tmp_path), "--min-axis", "4"]) == 0  # all skipped_valid
    assert main(["build", str(tmp_path / "x.bin")]) == 5


def test_cli_inspect_and_index(tmp_path, capsys):
    p = make_tiff(tmp_path / "s.tif", stack_u16(), compression="lzw", rowsperstrip=32)
    assert main(["inspect", str(p)]) == 0
    out = capsys.readouterr().out
    assert "tiff" in out and "servable_2d" in out and "transcode: yes" in out and "factor [1, 2, 2]" in out
    assert main(["inspect", str(p), "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert len(j["plan"][0]["levels"]) == 1  # 100x120 is below min_axis: nothing to build
    import os
    os.environ["ZB_BUILD_MIN_AXIS"] = "4"
    try:
        assert main(["inspect", str(p), "--json"]) == 0
    finally:
        del os.environ["ZB_BUILD_MIN_AXIS"]
    j = json.loads(capsys.readouterr().out)
    assert j["plan"][0]["levels"][1]["shape"] == [5, 50, 60] and j["estimated_cache_bytes"] > 0
    assert main(["index", str(p)]) == 5  # refkv rejects plain stacks
    o = make_tiff(tmp_path / "o.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    assert main(["index", str(o)]) == 0
    assert (tmp_path / ".zarrbridge" / "o.ome.tif" / "index.vzip").exists()


def test_cli_prune_orphan_central(tmp_path, capsys):
    p = make_mrc(tmp_path / "data" / "a.mrc", (16, 16, 16))
    central = tmp_path / "central"
    assert main(["--cache-mode", "central", "--central-root", str(central), "build", str(p), "--min-axis", "4"]) == 0
    caches = [d for d in central.rglob("fingerprint.json")]
    assert len(caches) == 1 and json.loads((caches[0].parent / "source.json").read_text())["path"] == str(p.resolve())
    p.unlink()
    assert main(["--cache-mode", "central", "--central-root", str(central), "prune", str(central)]) == 0
    assert not caches[0].exists()
