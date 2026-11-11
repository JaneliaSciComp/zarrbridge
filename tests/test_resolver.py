import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.conftest import make_mrc
from zarrbridge.dataset import Dataset
from zarrbridge.readers.base import IndexTimeout
from zarrbridge.server.config import Config
from zarrbridge.server.resolver import DatasetResolver, StillOpening


def _cfg(tmp_path, **limits):
    return Config.load(env={}, overrides={"roots": {"d": str(tmp_path)}, "build": {"auto": False}, "limits": limits})


def _slow_open(delay, calls):
    real = Dataset.open

    def open_(cls, path, config):
        calls.append(path)
        time.sleep(delay)
        return real(path, config)
    return classmethod(open_)


def test_concurrent_dataset_calls_open_once(tmp_path, monkeypatch):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    calls = []
    monkeypatch.setattr(Dataset, "open", _slow_open(0.3, calls))
    r = DatasetResolver(_cfg(tmp_path))
    out = []
    ts = [threading.Thread(target=lambda: out.append(r.dataset(p))) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(calls) == 1 and len(out) == 4 and all(o is out[0] for o in out)


def test_dataset_with_wait_reports_still_opening(tmp_path, monkeypatch):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    monkeypatch.setattr(Dataset, "open", _slow_open(0.5, []))
    r = DatasetResolver(_cfg(tmp_path))
    with pytest.raises(StillOpening) as e:
        r.dataset(p, wait=0.05)
    assert r.open_state(p)[0] == "indexing" and r.open_state(p)[1]["started_at"] == e.value.started_at
    assert r.dataset(p) is r.cached(p) and r.open_state(p) is None


def test_dataset_async_holds_no_thread(tmp_path, monkeypatch):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    monkeypatch.setattr(Dataset, "open", _slow_open(0.4, []))
    r = DatasetResolver(_cfg(tmp_path))

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(1))
        waits = [asyncio.create_task(r.dataset_async(p)) for _ in range(20)]
        await asyncio.sleep(0.05)
        t0 = loop.time()
        await loop.run_in_executor(None, lambda: None)   # the one pool thread is free: nobody is parked in it
        assert loop.time() - t0 < 0.2
        got = await asyncio.gather(*waits)
        assert all(g is got[0] for g in got)
        r._lru.clear()
        with pytest.raises(StillOpening):
            await r.dataset_async(p, wait=0.05)
    asyncio.run(main())


def test_unindexed_is_retried_after_open_retry_s(tmp_path, monkeypatch):
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    calls = []
    real = Dataset.open

    def open_(cls, path, config):
        calls.append(path)
        if len(calls) == 1:
            raise IndexTimeout(path, 60)
        return real(path, config)
    monkeypatch.setattr(Dataset, "open", classmethod(open_))
    r = DatasetResolver(_cfg(tmp_path, open_retry_s=0.2))
    with pytest.raises(IndexTimeout):
        r.dataset(p)
    state, extra = r.open_state(p)
    assert state == "unindexed" and "indexed manually" in extra["reason"]
    with pytest.raises(IndexTimeout):
        r.dataset(p)          # inside the retry window: no new open
    assert len(calls) == 1
    time.sleep(0.25)
    assert r.dataset(p) is not None and len(calls) == 2 and r.open_state(p) is None


def test_waiter_on_closed_loop_does_not_strand_others(tmp_path, monkeypatch):
    # A _status poll registers a waiter and times out; its loop may be gone when the open
    # ends. Notifying it must not raise and must not stop later waiters from being resolved.
    p = make_mrc(tmp_path / "a.mrc", (16, 16, 16))
    monkeypatch.setattr(Dataset, "open", _slow_open(0.6, []))
    r = DatasetResolver(_cfg(tmp_path))

    async def poll_and_leave():
        with pytest.raises(StillOpening):
            await r.dataset_async(p, wait=0.05)
    asyncio.run(poll_and_leave())            # this loop is now closed

    async def wait_for_it():
        return await asyncio.wait_for(r.dataset_async(p), 5)
    ds = asyncio.run(wait_for_it())
    assert ds is r.cached(p)
    op = r._opening.get(p)
    assert op is None
