import os
import socket
from pathlib import Path

from tests.conftest import make_mrc
from zarrbridge.cache.fingerprint import build_fingerprint, validate, write_fingerprint
from zarrbridge.cache.layout import cache_dir_for
from zarrbridge.cache.locks import BuildLock, is_locked, lock_info
from zarrbridge.readers.mrc.reader import MrcReader

PLAN = {"factor": [2, 2, 2], "levels": 2, "level_chunks": [[64, 64, 64]], "codec": "zstd:3", "method": "mean"}


def test_validate_states(tmp_path, cfg):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    op = MrcReader().open(p, cfg)
    d = tmp_path / "c"
    d.mkdir()
    assert validate(d, op) == ("missing", None)
    fp = build_fingerprint(op, PLAN, built_by="t", duration_s=1.0)
    assert fp["reader"] == {"name": "mrc", "version": 1} and fp["source"]["size"] == p.stat().st_size
    write_fingerprint(d, fp)
    assert validate(d, op) == ("valid", None)
    write_fingerprint(d, dict(fp, schema_version=99))
    assert validate(d, op) == ("incompatible", "schema_version")
    write_fingerprint(d, dict(fp, derivation_version=99))
    assert validate(d, op) == ("outdated", "derivation_version")
    write_fingerprint(d, dict(fp, reader={"name": "mrc", "version": 2}))
    assert validate(d, op) == ("outdated", "reader.version")
    write_fingerprint(d, fp)
    with open(p, "ab") as f:
        f.write(b"\0")
    reopened = MrcReader().open(p, cfg)
    assert validate(d, reopened) == ("stale", "size")
    lock = BuildLock.acquire(tmp_path / "c2")
    assert validate(tmp_path / "c2", op) == ("building", None)
    lock.release()


def test_lock_contention(tmp_path):
    a = BuildLock.acquire(tmp_path)
    assert a is not None and BuildLock.acquire(tmp_path) is None and is_locked(tmp_path)
    host, pid, _ = lock_info(tmp_path)
    assert host == socket.gethostname() and pid == os.getpid()
    a.release()
    assert not is_locked(tmp_path)


def test_stale_lock_broken(tmp_path):
    (tmp_path / "lock").write_text(f"{socket.gethostname()}:999999:2000-01-01T00:00:00Z")
    os.utime(tmp_path / "lock", (0, 0))
    # not flocked by anyone -> acquire succeeds directly; a flocked-but-dead-pid lock is exercised via subprocess in the builder tests
    assert BuildLock.acquire(tmp_path, stale_seconds=1) is not None


def test_cache_dir_modes(tmp_path, cfg):
    assert cache_dir_for(tmp_path / "a.mrc", cfg) == (tmp_path / ".zarrbridge" / "a.mrc", "beside")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        d, m = cache_dir_for(ro / "a.mrc", cfg)
        if os.access(ro, os.W_OK):  # running as root: cannot test read-only fallback
            return
        assert m == "central" and str(d).startswith(os.path.expanduser("~/.cache/zarrbridge/pyramids"))
    finally:
        ro.chmod(0o755)
