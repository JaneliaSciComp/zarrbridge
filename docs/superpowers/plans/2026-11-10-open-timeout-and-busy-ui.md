# Bounded opens and a busy UI for zarrbridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** zarrbridge gives up indexing a file after 60 s instead of working for an hour in the background, reports what it is doing while it works, says clearly when a file is too big to index automatically and that it must be indexed manually, and Fileglancer shows a working animation whenever zarrbridge is busy on the file being viewed.

**Architecture:** The refkv zarrify is run as a cancellable coroutine with a deadline that only the server passes (the builder and the `index` CLI stay unbounded); a timeout raises a new `IndexTimeout`, which the reader registry re-raises rather than falling through to the tifffile reader. Routes wait on the resolver through `asyncio.to_thread`, so the event loop keeps serving other files while one is opening. The dataset resolver opens files on a background thread, dedupes concurrent opens, remembers timeouts for a short while, and lets `_status` answer without blocking, so the status endpoint reports `indexing` during the open and the new terminal state `unindexed` after a timeout. Fileglancer's `virtual_node` consults `_status` first and returns a pending root node, its catch-all stops turning transport errors into "plain file", and the frontend drives everything off `levels.state`: a spinner for the three transient states, warning copy for `unindexed`, and a listing refetch when the state leaves a transient value.

**Tech Stack:** zarrbridge (Python 3.12, Starlette, refkv 0.2.0, pytest via `pixi run -e test test`); fileglancer backend (FastAPI, httpx, pytest); fileglancer frontend (React 18, TanStack Query v5, Vitest + Testing Library via `pixi run test-frontend`).

**Spec:** The request of 2026-11-10 in the fileglancer session, quoted here as the requirements: (1) "I don't want zarrbridge to be zarrifying something for minutes in the background. It should fail quickly (60s)." (2) "While it's running, the UI should be clear about it processing." (3) "When it fails, it should be clear about the error - 'this is too big to zarrify automatically', and the resolution - 'this needs to be run manually'. We will add a Fileglancer App for this later." (4) "We need to update the UI so that there is a working animation any time something is happening in the background with zarrbridge." Background findings are in `notes/2026-11-10-imagej-truncated-tiff.md` and the session: a 1.86 TB CZI with 252,780 subblocks takes about an hour to zarrify because refkv reads every subblock header; fileglancer's 60 s httpx timeout then shows the file as a plain file while the dataserver keeps reading.

## Global Constraints

- Every command runs through pixi. zarrbridge tests: `pixi run -e test test <files>` from `../zarrbridge`. fileglancer backend tests: `.pixi/envs/test/bin/python -m pytest <files>` from the fileglancer root (the `test-backend` task's pip step is broken locally). Frontend tests: `pixi run test-frontend -- src/__tests__/...` (paths relative to `frontend/`).
- Commit each task with explicit paths, never `git add -A`. End every commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never push.
- zarrbridge work goes on branch `status-states`; fileglancer work on branch `zarrbridge`. The fileglancer dev server uses the editable zarrbridge install, so no release is needed to test by hand.
- State names are the contract between the two repos: transient `indexing | queued | building`; terminal `native | multiscale | failed | unservable | unindexed`. Only the transient three are ever polled.
- The default deadline is 60 s (`limits.open_timeout_s = 60.0`). The timeout reason text must contain both phrases the request asks for: "too big to index automatically" and "indexed manually".
- Copy in the UI is plain English, no jargon beyond "OME-Zarr" and "zarrbridge"; file paths and commands appear in the zarrbridge-provided reason, not in frontend string literals.
- No new dependencies in either repo.

## Review Focus

- A dataserver that dies or restarts while a root listing is in flight: the user must see an error, never the hex dump of the image file. Pinned by Task 5's `test_transport_error_is_503_not_plain_file`.
- A user who reloads the page during the 60 s window: no second zarrify of the same file may start. Pinned by Task 2's `test_concurrent_dataset_calls_open_once`.
- A fast file (MRC, small TIFF) must not flash "indexing" on first visit: `_status` waits a short grace period for the open before answering pending. Pinned by Task 3's `test_status_of_fast_file_is_complete_on_first_request`.
- A timeout followed by a manual `zarrbridge index`: the server must pick up the saved index after the retry window, not stay `unindexed` forever. Pinned by Task 2's `test_unindexed_is_retried_after_open_retry_s`.
- Indexing finishes between the pending root listing and the first status poll: the listing must still be refreshed. Pinned by Task 7's `shouldRefreshListing` test.
- A dataset opening for 60 s must not stall chunk requests for other files on the same dataserver. Pinned by Task 3's `test_pending_open_does_not_block_other_requests`.
- `zarrbridge build` and `zarrbridge index` must run to completion on a file the server gave up on; they are the manual resolution. Pinned by Task 1's `test_timeout_is_opt_in`.
- The `_status` key is reachable from inside the hierarchy (`image.tif/0/_status`) and the pending answer must be the same there. Pinned by Task 3's `test_status_pending_shape` which calls both forms.

---

### Task 1: A cancellable, time-bounded zarrify

**Files:**
- Modify: `zarrbridge/_sync.py`
- Modify: `zarrbridge/readers/base.py`
- Modify: `zarrbridge/server/config.py:105-116` (`LimitsConfig`)
- Modify: `zarrbridge/readers/refkv_reader.py:40-92` (`RefkvReader.open`, `_zarrify`)
- Modify: `zarrbridge/readers/registry.py:36-66` (`open_dataset_reader`)
- Modify: `zarrbridge/dataset.py:53` (the one `open_dataset_reader` call)
- Test: `tests/test_refkv_reader.py`

**Interfaces:**
- Produces: `run_sync_with_timeout(coro: Awaitable[T], timeout: float | None) -> T` in `zarrbridge/_sync.py`; raises builtin `TimeoutError` after cancelling the coroutine's task on the background loop. What cancellation buys, and what it does not: refkv's `drive_async` awaits each read batch through its own thread pool and shuts that pool down with `cancel_futures=True` in a `finally`, so at most `readers.refkv.connections` in-flight reads finish and no further read is scheduled. The projection step (`asyncio.to_thread(output)`) runs to completion if cancellation lands during it; it is CPU work on data already read, seconds for this file. The hour-long read walk is what stops. Process isolation would make the stop absolute and is not worth a subprocess per open; say so in the spec.
- Produces: `class IndexTimeout(ReaderRejected)` in `zarrbridge/readers/base.py` with `__init__(self, path: Path, timeout_s: float)` and attribute `timeout_s`; its message is `f"indexing did not finish within {timeout_s:.0f} s, so this file is too big to index automatically. It needs to be indexed manually: run `zarrbridge index {path}`; the saved index makes later opens fast."`.
- Produces: `LimitsConfig.open_timeout_s: float = 60.0`, `LimitsConfig.open_grace_s: float = 2.0`, `LimitsConfig.open_retry_s: float = 60.0`, `LimitsConfig.index_save_after_s: float = 5.0`. Zero disables each.
- Produces: `_zarrify(path: Path, config, timeout: float | None) -> ReferenceStore`; `None` means no deadline; raises `IndexTimeout` otherwise.
- Produces: `RefkvReader.open(path, config, cache_dir=None, *, timeout: float | None = None)` and `open_dataset_reader(path, config, cache_dir=None, *, timeout: float | None = None)`, both threading `timeout` down to `_zarrify`. The default is unbounded, so `builder.build` (`builder.py:87`) and `cli._index` (`cli.py:206`) are untouched and keep running to completion. Only `Dataset.open` passes `config.limits.open_timeout_s or None`; `zarrbridge status` goes through `Dataset.open` and so prints the timeout reason for a file the server would give up on.
- Produces: `open_dataset_reader` re-raises `IndexTimeout` from the refkv attempt instead of recording it and trying the tifffile reader; a timeout is final.

- [ ] **Step 1: Write the failing test**

```python
def test_zarrify_timeout_cancels_and_raises(tmp_path, cfg, monkeypatch):
    import asyncio, time
    import refkv.zarrify
    from zarrbridge.readers.base import IndexTimeout
    from zarrbridge.readers.refkv_reader import _zarrify
    state = {"cancelled": False}

    async def slow(*a, **k):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    monkeypatch.setattr(refkv.zarrify, "zarrify_async", slow)
    t0 = time.time()
    with pytest.raises(IndexTimeout, match="too big to index automatically"):
        _zarrify(tmp_path / "x.czi", cfg, timeout=0.2)
    assert time.time() - t0 < 2
    time.sleep(0.1)
    assert state["cancelled"]


def test_cancel_stops_scheduling_executor_work(tmp_path, cfg, monkeypatch):
    # Cancellation lands while a read is running on refkv's executor: that read
    # finishes, nothing after it starts. This is the guarantee the spec states.
    import asyncio, time
    import refkv.zarrify
    from zarrbridge.readers.base import IndexTimeout
    from zarrbridge.readers.refkv_reader import _zarrify
    steps = []

    def blocking_read(i):
        time.sleep(0.05)
        steps.append(i)

    async def fake(*a, **k):
        loop = asyncio.get_running_loop()
        for i in range(100):
            await loop.run_in_executor(None, blocking_read, i)

    monkeypatch.setattr(refkv.zarrify, "zarrify_async", fake)
    with pytest.raises(IndexTimeout):
        _zarrify(tmp_path / "x.czi", cfg, timeout=0.3)
    n = len(steps)
    assert 3 <= n <= 8          # a few reads happened, then the deadline
    time.sleep(0.4)
    assert len(steps) <= n + 1  # at most the in-flight read completed afterwards


def test_timeout_is_opt_in(tmp_path, cfg, monkeypatch):
    # The builder and `zarrbridge index` never pass a timeout: they are the manual resolution.
    import time
    import zarrbridge.readers.refkv_reader as rr
    from zarrbridge.build.builder import build, BUILT
    real = rr._zarrify

    def slow(path, config, timeout):
        assert timeout is None
        time.sleep(0.3)
        return real(path, config, timeout)
    monkeypatch.setattr(rr, "_zarrify", slow)
    cfg.limits.open_timeout_s = 0.05
    cfg.build.min_axis = 4
    p = make_tiff(tmp_path / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    assert build(p, cfg) == BUILT
    assert RefkvReader().open(p, cfg).identity.extras["format"] == "tiff"


def test_cli_index_ignores_open_timeout(tmp_path, monkeypatch):
    # `zarrbridge index` is the recovery command the unindexed reason names: it must not time out.
    import time
    import zarrbridge.readers.refkv_reader as rr
    from zarrbridge.cli import main
    real = rr._zarrify

    def slow(path, config, timeout):
        assert timeout is None
        time.sleep(0.3)
        return real(path, config, timeout)
    monkeypatch.setattr(rr, "_zarrify", slow)
    monkeypatch.setenv("ZB_LIMITS_OPEN_TIMEOUT_S", "0.05")  # whatever spelling tests/test_config.py shows for a nested limits field
    p = make_tiff(tmp_path / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    assert main(["--cache-mode", "central", "--central-root", str(tmp_path / "c"), "index", str(p)]) == 0
    assert list((tmp_path / "c").rglob("index.vzip"))


def test_registry_does_not_swallow_timeout(tmp_path, cfg, monkeypatch):
    import zarrbridge.readers.refkv_reader as rr
    from zarrbridge.readers.base import IndexTimeout
    monkeypatch.setattr(rr, "_zarrify", lambda path, config, timeout: (_ for _ in ()).throw(IndexTimeout(path, 60)))
    p = make_tiff(tmp_path / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    with pytest.raises(IndexTimeout):
        open_dataset_reader(p, cfg, timeout=60)
    # no timeout requested: not even attempted, the real zarrify runs
    monkeypatch.undo()
    assert open_dataset_reader(p, cfg).reader_name == "refkv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e test test tests/test_refkv_reader.py::test_zarrify_timeout_cancels_and_raises`
Expected: FAIL with `ImportError: cannot import name 'IndexTimeout'`

Run: `pixi run -e test test tests/test_refkv_reader.py -k "opt_in or swallow or cli_index or cancel_stops"`
Expected: FAIL with `TypeError` (unexpected keyword `timeout`) and `ImportError` for `IndexTimeout`

- [ ] **Step 3: Implement `run_sync_with_timeout` in `zarrbridge/_sync.py`**

Schedule with `asyncio.run_coroutine_threadsafe(coro, _bg_loop())`, call `.result(timeout)`; on `concurrent.futures.TimeoutError` call `fut.cancel()` and raise `TimeoutError`. Always use the background loop, even when no loop runs in the caller's thread, so cancellation has a loop to run on.

- [ ] **Step 4: Add `IndexTimeout` to `readers/base.py` and the four limits to `LimitsConfig`**

- [ ] **Step 5: Rewrite `_zarrify(path, config, timeout)` to use `refkv.zarrify.zarrify_async` through `run_sync_with_timeout`**

Translate `TimeoutError` into `IndexTimeout(path, timeout)`. Import `zarrify_async` from the module at call time (`refkv.zarrify.zarrify_async`) so the test's monkeypatch is seen. Thread the keyword through `RefkvReader.open` and `open_dataset_reader`, defaulting to `None`.

- [ ] **Step 6: In `open_dataset_reader`, re-raise `IndexTimeout` inside the TIFF reader loop before the `except ReaderRejected` branch records it**

- [ ] **Step 7: In `Dataset.open`, pass `timeout=config.limits.open_timeout_s or None`**

- [ ] **Step 8: Run the reader, builder and dataset tests**

Run: `pixi run -e test test tests/test_refkv_reader.py tests/test_builder.py tests/test_dataset.py tests/test_cli.py`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add zarrbridge/_sync.py zarrbridge/readers/base.py zarrbridge/server/config.py zarrbridge/readers/refkv_reader.py zarrbridge/readers/registry.py zarrbridge/dataset.py tests/test_refkv_reader.py
git commit -m "feat: give up zarrifying after limits.open_timeout_s"
```

---

### Task 2: Resolver opens in the background, dedupes, and remembers timeouts

**Files:**
- Modify: `zarrbridge/server/resolver.py:28-120`
- Test: `tests/test_resolver.py` (new)

**Interfaces:**
- Consumes: `IndexTimeout`, `LimitsConfig.open_timeout_s`, `open_retry_s` from Task 1.
- Produces: `class StillOpening(Exception)` with attributes `started_at: float` (epoch seconds).
- Produces: `DatasetResolver.dataset(path: Path, wait: float | None = None) -> Dataset`, the blocking form for the CLI and tests. Starts an open if none is in flight, joins the in-flight one otherwise, waits up to `wait` seconds (None: until the open ends, which the Task 1 deadline bounds), then returns the dataset, or raises `StillOpening`, `IndexTimeout`, or `ReaderRejected`.
- Produces: `async def DatasetResolver.dataset_async(path: Path, wait: float | None = None) -> Dataset`, the same contract for the routes. It holds no thread while waiting: each `_Opening` keeps `waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]]`; `dataset_async` registers a future and awaits it under `asyncio.wait_for(..., wait)`; the opener thread, after recording the outcome, calls `loop.call_soon_threadsafe(fut.set_result, None)` for every waiter. Only `Dataset.open` itself runs on a thread.
- Produces: `DatasetResolver.cached(path: Path) -> Dataset | None` (LRU lookup, no open, no source check).
- Produces: `DatasetResolver.open_state(path: Path) -> tuple[str, dict] | None`: `("indexing", {"started_at": float})` while an open is in flight; `("unindexed", {"reason": str})` for `open_retry_s` after an `IndexTimeout`; `("unservable", {"reason": str})` for the existing 30 s after a `ReaderRejected`; else None.
- Keeps: `open_count`, `close_idle`, `root_dir`, `split`, `check_trusted` unchanged. `ensure_pyramid("auto", queue)` still runs once per successful open.

- [ ] **Step 1: Write the failing tests**

```python
import threading, time
from pathlib import Path
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
    [t.start() for t in ts]; [t.join() for t in ts]
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
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e test test tests/test_resolver.py`
Expected: FAIL with `ImportError: cannot import name 'StillOpening'`

- [ ] **Step 3: Implement the in-flight table in `DatasetResolver`**

Add `self._opening: dict[Path, _Opening]` where `_Opening` is a small class with `started_at: float`, `done: threading.Event`, `result: Dataset | None`, `error: BaseException | None`. `dataset()` under `self._lock`: return the LRU hit as today; otherwise find or create the `_Opening`, and if created start `threading.Thread(target=self._open, args=(path, opening), name=f"zarrbridge-open", daemon=True)`. Outside the lock wait `opening.done.wait(wait)`; not done raises `StillOpening(opening.started_at)`; done re-raises `opening.error` or returns `opening.result`. `_open` runs `Dataset.open(path, self.config)`, then under the lock inserts into the LRU (closing a racing duplicate as the current code does), evicts past `max_open_datasets`, records `IndexTimeout` in `self._unindexed[path] = (str(e), time.time())` and `ReaderRejected` in `self._rejected` as today, deletes the `_Opening`, and sets `done`. Call `ensure_pyramid("auto", self.queue)` after insertion when `build.auto`. Before starting an open, consult `_unindexed` within `open_retry_s` and `_rejected` within 30 s and raise the remembered error instead.

- [ ] **Step 4: Implement `dataset_async`, `cached` and `open_state`**

`dataset_async` shares the start-or-join step with `dataset` (factor it into `_start_or_join(path) -> _Opening | Dataset`), then awaits its future instead of `done.wait`. Remembered errors (`_unindexed`, `_rejected`) are raised before anything is scheduled.

- [ ] **Step 5: Run the resolver and server tests**

Run: `pixi run -e test test tests/test_resolver.py tests/test_server.py`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add zarrbridge/server/resolver.py tests/test_resolver.py
git commit -m "feat: open datasets in the background, once per path, and remember timeouts"
```

---

### Task 3: `_status` never blocks; `unindexed` is a state; spec text

**Files:**
- Modify: `zarrbridge/server/app.py:157-200` (`status`, `key_handler`)
- Modify: `zarrbridge/dataset.py:177-199` (`status` cache block already has `writable`; no change unless needed)
- Modify: `docs/specs/zarrbridge-spec.md:261-292` (§8.4, §8.6)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `resolver.dataset_async(path, wait=...)`, `resolver.open_state(path)`, `StillOpening`, `IndexTimeout` from Tasks 1 and 2. Both routes are `async def` and await `dataset_async`; no route calls the blocking `dataset`, and no thread is parked while a request waits for an open.
- Produces: the pending status JSON, status 200: `{"path": str, "reader": None, "reader_reason": None, "classification": "pending", "images": [], "levels": {"state": "indexing", "built": 0, "planned": None, "progress": 0.0, "eta_s": None, "reason": "reading the file's structure; gives up after {open_timeout_s:.0f} s", "elapsed_s": float}, "cache": {"mode": str, "dir": str, "bytes": 0, "fingerprint": "missing", "mismatch": None, "writable": bool}, "advice": []}`.
- Produces: the unindexed status JSON, status 200: same shape with `"classification": "unindexed"`, `"levels": {"state": "unindexed", "built": 0, "planned": None, "progress": 0.0, "eta_s": None, "reason": <IndexTimeout message>}`.
- Produces: `key_handler` answers `422` with body `unindexed: <message>` for a timed-out file, as it does `unservable: ...` today.

- [ ] **Step 1: Write the failing tests**

```python
def _slow_zarrify(monkeypatch, delay):
    import zarrbridge.readers.refkv_reader as rr
    real = rr._zarrify
    calls = []

    def z(path, config):
        calls.append(path)
        time.sleep(delay)
        return real(path, config)
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
    assert "gives up after" in s["levels"]["reason"] and s["cache"]["mode"] == "beside"
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
    monkeypatch.setattr(rr, "_zarrify", lambda path, config: (_ for _ in ()).throw(IndexTimeout(path, 60)))
    c = TestClient(create_app(cfg_root("data", tmp_path / "data")))
    s = c.get("/_status/data/a.ome.tif").json()
    assert s["classification"] == "unindexed" and s["levels"]["state"] == "unindexed"
    assert "too big to index automatically" in s["levels"]["reason"] and "indexed manually" in s["levels"]["reason"]
    r = c.get("/data/a.ome.tif/zarr.json")
    assert r.status_code == 422 and r.text.startswith("unindexed:")


def test_pending_open_does_not_block_other_requests(tmp_path, monkeypatch):
    make_tiff(tmp_path / "data" / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    make_mrc(tmp_path / "data" / "b.mrc", (16, 16, 16))
    _slow_zarrify(monkeypatch, 1.5)
    c = TestClient(create_app(cfg_root("data", tmp_path / "data")))
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e test test tests/test_server.py -k "pending_shape or fast_file or unindexed or does_not_block"`
Expected: FAIL (`pending_shape` blocks and asserts on elapsed time; `unindexed` gets `unservable`; `does_not_block` exceeds 0.5 s because the loop is held by the open)

- [ ] **Step 3: Rewrite the `status` route**

Call `await resolver.dataset_async(path, wait=config.limits.open_grace_s)`. On `StillOpening` build the pending JSON (`elapsed_s = time.time() - e.started_at`, cache fields from `cache_dir_for(path, config)` and `cache_writable`). On `IndexTimeout` build the unindexed JSON. Keep the `ReaderRejected` branch as it is. Note `IndexTimeout` is a `ReaderRejected`, so catch it first.

- [ ] **Step 4: Update `key_handler`**

`await resolver.dataset_async(path)` with no wait (the Task 1 deadline bounds it); catch `IndexTimeout` before `ReaderRejected` and return `PlainTextResponse(f"unindexed: {e}", status_code=422)`.

- [ ] **Step 5: Update the spec**

§8.4: add `unindexed` to the state set with one sentence: "`unindexed` is terminal: the open hit `limits.open_timeout_s`; the file is too big to index automatically and must be indexed manually with `zarrbridge index`, after which the saved index makes the next open fast." Add: "`_status` never blocks for longer than `limits.open_grace_s`; while a dataset is being opened it reports `indexing` with `elapsed_s`. Opens run on worker threads; requests waiting for one hold neither the loop nor a thread." In §8.6 under `open_timeout_s`: "cancellation stops the read walk (at most `readers.refkv.connections` in-flight reads finish); a projection already running completes." Also: "`limits.open_timeout_s` applies to the server only. `zarrbridge build` and `zarrbridge index` have no deadline: they are how a file the server gave up on gets indexed." §8.6: add `limits.open_timeout_s` (60 s; the read walk is cancelled), `open_grace_s` (2 s), `open_retry_s` (60 s; a timed-out file is not reopened sooner), `index_save_after_s` (5 s; see Task 4).

- [ ] **Step 6: Run the whole suite**

Run: `pixi run -e test test`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add zarrbridge/server/app.py docs/specs/zarrbridge-spec.md tests/test_server.py
git commit -m "feat: non-blocking _status with indexing and unindexed states"
```

---

### Task 4: The server saves the refkv index after a slow open

**Files:**
- Modify: `zarrbridge/dataset.py:50-56` (`Dataset.open`)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `has_unsaved_index(opened)`, `save_index(opened, cache_dir, binding)` from `zarrbridge/readers/refkv_reader.py`; `cache_writable` from `zarrbridge/cache/layout.py`; `LimitsConfig.index_save_after_s` from Task 1.
- Produces: `Dataset.open` saves the index when the reader open took at least `index_save_after_s` seconds, the reader has an unsaved index, and the cache dir is writable. Failures are logged at warning and ignored.

- [ ] **Step 1: Write the failing test**

```python
def test_slow_open_saves_index(tmp_path, cfg):
    from tests.conftest import make_tiff, stack_u16
    p = make_tiff(tmp_path / "a.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32))
    cfg.limits.index_save_after_s = 0
    ds = Dataset.open(p, cfg)
    assert (ds.cache_dir / "index.json").exists() and any(ds.cache_dir.glob("index.vzip"))
    ds.close()
    cfg.limits.index_save_after_s = 3600
    ds2 = Dataset.open(make_tiff(tmp_path / "b.ome.tif", stack_u16(), ome=True, metadata={"axes": "ZYX"}, tile=(32, 32)), cfg)
    assert not (ds2.cache_dir / "index.json").exists()
    ds2.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e test test tests/test_dataset.py::test_slow_open_saves_index`
Expected: FAIL on the first `exists()` assertion

- [ ] **Step 3: Implement in `Dataset.open`**

Time `open_dataset_reader`; afterwards, when `elapsed >= config.limits.index_save_after_s and has_unsaved_index(opened) and cache_writable(cache_dir)`, call `save_index(opened, cache_dir, config.readers.refkv.index_binding)` inside `try/except Exception` with `_log.warning`. Log at info when saved, including the elapsed seconds.

- [ ] **Step 4: Run the dataset tests**

Run: `pixi run -e test test tests/test_dataset.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add zarrbridge/dataset.py tests/test_dataset.py
git commit -m "feat: save the refkv index after a slow server-side open"
```

---

### Task 5: Fileglancer backend: pending root node, no "plain file" on transport errors

**Files:**
- Modify: `fileglancer/zarrproxy.py:129-163` (`virtual_node`)
- Modify: `fileglancer/server.py:603-625` (`_virtual_node`)
- Modify: `frontend/src/shared.types.ts:17-19` (`FileOrFolder`)
- Test: `tests/test_zarrproxy.py`

**Interfaces:**
- Consumes: zarrbridge `_status` shapes from Task 3.
- Produces: `zarrproxy.PENDING_STATES = frozenset({"indexing", "unindexed"})`.
- Produces: `virtual_node(sock, abs_path, info, key="")` at the root first GETs `/_status`; when it is 200 and `levels.state in PENDING_STATES` it returns `{"info": {**info, "virtual_zarr": True, "zarrbridge_state": state}, "files": []}` without requesting `zarr.json`. Any other status answer falls through to today's logic. Below the root nothing changes.
- Produces: `virtual_node` raises `ValueError` only for a 4xx root `zarr.json` (zarrbridge saying the file is not servable: 404, 415, 422); any 5xx calls `resp.raise_for_status()`, which raises `httpx.HTTPStatusError`, an `httpx.HTTPError`.
- Produces: `_virtual_node` maps `ValueError` to the plain file as today, and `httpx.HTTPError`, `WorkerDead`, `WorkerError` to `HTTPException(status_code=503, detail=f"zarrbridge unavailable: {e}")`.
- Produces: `FileOrFolder.zarrbridge_state?: string` in `shared.types.ts`.

- [ ] **Step 1: Write the failing tests**

```python
def _mock_client(monkeypatch, handler):
    import httpx
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://zarrbridge")
    monkeypatch.setattr(zarrproxy, "_client", lambda sock: client)


@pytest.mark.asyncio
async def test_virtual_node_root_while_indexing(monkeypatch):
    import httpx
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if req.url.path == "/_status":
            return httpx.Response(200, json={"classification": "pending", "levels": {"state": "indexing"}})
        return httpx.Response(200, json={"zarr_format": 3, "node_type": "group"})
    _mock_client(monkeypatch, handler)
    info = {"name": "a.czi", "path": "d/a.czi", "is_dir": False, "size": 1}
    node = await zarrproxy.virtual_node("/tmp/x.sock", "/abs/d/a.czi", info)
    assert node == {"info": {**info, "virtual_zarr": True, "zarrbridge_state": "indexing"}, "files": []}
    assert seen == ["/_status"]


# added to the existing `TestVirtualZarr` class
    def test_transport_error_is_503_not_plain_file(self, zarr_app, monkeypatch):
        import httpx
        _mock_client(monkeypatch, lambda req: (_ for _ in ()).throw(httpx.ConnectError("gone")))
        r = zarr_app.get("/api/files/share", params={"subpath": "data/tiny.tif"})
        assert r.status_code == 503 and "zarrbridge unavailable" in r.text

    def test_upstream_5xx_is_503_not_plain_file(self, zarr_app, monkeypatch):
        import httpx

        def handler(req):
            if req.url.path == "/_status":
                return httpx.Response(200, json={"levels": {"state": "native"}})
            return httpx.Response(503, text="worker restarting")
        _mock_client(monkeypatch, handler)
        r = zarr_app.get("/api/files/share", params={"subpath": "data/tiny.tif"})
        assert r.status_code == 503

    def test_upstream_4xx_is_plain_file(self, zarr_app, monkeypatch):
        import httpx
        _mock_client(monkeypatch, lambda req: httpx.Response(422, text="unservable: no reader"))
        r = zarr_app.get("/api/files/share", params={"subpath": "data/tiny.tif"})
        assert r.status_code == 200 and r.json()["info"].get("virtual_zarr") is None and "files" not in r.json()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pixi/envs/test/bin/python -m pytest tests/test_zarrproxy.py -k "while_indexing or transport_error or upstream"`
Expected: FAIL (`seen` includes `/zarr.json`; the 503 cases return 200 with the plain file; the 4xx case passes already)

- [ ] **Step 3: Implement `PENDING_STATES` and the root pre-check in `virtual_node`**

- [ ] **Step 4: In `virtual_node`, call `resp.raise_for_status()` when `resp.status_code >= 500` before the existing `ValueError`; narrow the catch-all in `_virtual_node`**

Import `httpx` in `server.py` if not already imported; the `WorkerDead` and `WorkerError` names are already imported there (line 49).

- [ ] **Step 5: Add `zarrbridge_state?: string` to `FileOrFolder`**

- [ ] **Step 6: Run the zarrproxy tests**

Run: `.pixi/envs/test/bin/python -m pytest tests/test_zarrproxy.py`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add fileglancer/zarrproxy.py fileglancer/server.py frontend/src/shared.types.ts tests/test_zarrproxy.py
git commit -m "fix: show zarrbridge indexing as a pending node and transport errors as 503"
```

---

### Task 6: Frontend status helpers and hint copy with a spinner

**Files:**
- Modify: `frontend/src/queries/zarrQueries.ts:425-475`
- Modify: `frontend/src/components/ui/BrowsePage/MetadataHint.tsx`
- Test: `frontend/src/__tests__/unitTests/zarrbridgePolling.test.ts`
- Test: `frontend/src/__tests__/componentTests/MetadataHint.test.tsx` (new)

**Interfaces:**
- Produces: `export function isZarrbridgeBusy(state: string | undefined): boolean` (true for `indexing | queued | building`).
- Produces: `zarrbridgePollInterval` returns 2000 for `indexing`, 5000 for `queued | building`, false otherwise (unchanged for errors).
- Produces: `ZarrbridgeStatus.levels` gains `eta_s?: number | null; elapsed_s?: number`.
- Produces: `MetadataHintVariant` case `{ case: 'virtual-zarr'; state?: string; reason?: string | null; progress?: number }`; `HintConfig.busy?: boolean`.
- Produces: `virtualZarrHint(state, reason, progress): HintConfig` with this copy:
  - `indexing`: info, busy, title "Reading image structure", description "zarrbridge is indexing this image file so it can be served as OME-Zarr. This usually takes seconds and is given up after a minute."
  - `queued`: info, busy, title "Multiscale pyramid queued", description "A multiscale pyramid will be built in the background. Until it is done, viewers read full resolution at every zoom level."
  - `building`: info, busy, title "Building multiscale pyramid", description `${Math.round(progress * 100)}% done. Until it finishes, viewers read full resolution at every zoom level.` (omit the percentage sentence when `progress` is undefined).
  - `unindexed`: warning, title "Too large to index automatically", description `reason ?? 'Indexing did not finish in time, so this file must be indexed manually.'`
  - `failed` / `unservable`: warning, title "Virtual OME-Zarr", description as today.
  - `multiscale`: info, title "Virtual OME-Zarr", description as today.
  - `error`: warning, title "Virtual OME-Zarr status unavailable", description `reason ?? 'zarrbridge did not answer.'`
  - default (`native`): info, title "Virtual OME-Zarr", today's text followed by ` ${reason}` when `reason` is set.
- Produces: `MetadataHint` renders `<Spinner sizeClasses="w-4 h-4 border-2" />` from `@/components/ui/widgets/Loaders` to the left of the notification inside a `<div data-testid="zarrbridge-busy" className="flex items-start gap-3">` when `busy` is true.

- [ ] **Step 1: Extend the polling test and write the hint test**

In `zarrbridgePolling.test.ts` change the first test to expect 2000 for `indexing` and add `'unindexed'` to the terminal list. New file:

```tsx
import { render, screen } from '@/__tests__/test-utils';
import MetadataHint from '@/components/ui/BrowsePage/MetadataHint';

describe('MetadataHint virtual-zarr', () => {
  it('shows a spinner while zarrbridge is busy', () => {
    render(<MetadataHint variant={{ case: 'virtual-zarr', state: 'indexing' }} />);
    expect(screen.getByTestId('zarrbridge-busy')).toBeInTheDocument();
    expect(screen.getByText('Reading image structure')).toBeInTheDocument();
  });
  it('shows progress while building', () => {
    render(<MetadataHint variant={{ case: 'virtual-zarr', state: 'building', progress: 0.42 }} />);
    expect(screen.getByText(/42% done/)).toBeInTheDocument();
  });
  it('explains an unindexed file with zarrbridge\'s reason', () => {
    render(<MetadataHint variant={{ case: 'virtual-zarr', state: 'unindexed', reason: 'run zarrbridge index /x' }} />);
    expect(screen.getByText('Too large to index automatically')).toBeInTheDocument();
    expect(screen.getByText(/run zarrbridge index \/x/)).toBeInTheDocument();
    expect(screen.queryByTestId('zarrbridge-busy')).toBeNull();
  });
  it('has no spinner in a terminal state', () => {
    render(<MetadataHint variant={{ case: 'virtual-zarr', state: 'native' }} />);
    expect(screen.queryByTestId('zarrbridge-busy')).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run test-frontend -- src/__tests__/unitTests/zarrbridgePolling.test.ts src/__tests__/componentTests/MetadataHint.test.tsx`
Expected: FAIL (2000 vs 5000; no `zarrbridge-busy`; old titles)

- [ ] **Step 3: Implement `isZarrbridgeBusy`, the new poll intervals and the type fields in `zarrQueries.ts`**

- [ ] **Step 4: Replace `virtualZarrDescription` with `virtualZarrHint` and render the spinner in `MetadataHint.tsx`**

- [ ] **Step 5: Run the tests, then lint**

Run: `pixi run test-frontend -- src/__tests__/unitTests/zarrbridgePolling.test.ts src/__tests__/componentTests/MetadataHint.test.tsx` then `pixi run node-eslint-write && pixi run node-check`
Expected: tests PASS; no new type errors (the pre-existing ones listed in memory are not blocking)

- [ ] **Step 6: Commit**

```bash
git add frontend/src/queries/zarrQueries.ts frontend/src/components/ui/BrowsePage/MetadataHint.tsx frontend/src/__tests__/unitTests/zarrbridgePolling.test.ts frontend/src/__tests__/componentTests/MetadataHint.test.tsx
git commit -m "feat: spinner and clear copy for every zarrbridge state"
```

---

### Task 7: FileBrowser drives everything off the zarrbridge state

**Files:**
- Modify: `frontend/src/components/ui/BrowsePage/FileBrowser.tsx:101-125, 269-300, 365-376`
- Modify: `frontend/src/queries/zarrQueries.ts` (`shouldRefreshListing`)
- Test: `frontend/src/__tests__/unitTests/zarrbridgeRefresh.test.ts` (new)

**Interfaces:**
- Consumes: `isZarrbridgeBusy`, `useZarrbridgeStatusQuery` from Task 6; `FileOrFolder.virtual_zarr` and `zarrbridge_state` from Task 5.
- Produces: `isVirtualZarr = fileQuery.data?.currentFileOrFolder?.virtual_zarr === true` (no longer requires `isZarrDir`). The status query is enabled by `isVirtualZarr` alone.
- Produces: the hint renders when `isVirtualZarr && (zarrbridgeStatusQuery.data || zarrbridgeStatusQuery.isError)`, passing `state` (or `'error'` with `reason = error.message` on a failed query) and `progress`.
- Produces: `ZarrPreview` renders only when `isZarrDir && zarrMetadataQuery.data?.metadata`; `FileViewer` renders only when the item is a file, `!isZarrDir` and `!virtual_zarr`; when `isVirtualZarr && !isZarrDir` the main area below the hint is empty.
- Produces: one `useEffect` keyed on `[zarrbridgeStatusQuery.data?.levels?.state, fileQuery.data?.currentFileOrFolder?.zarrbridge_state]` that refreshes when `shouldRefreshListing(prevState, state, nodeState)` (below) is true, and does nothing otherwise:
  - **Reconcile** (covers a first status answer that is already terminal): the listing's node carries a `zarrbridge_state` (it was returned pending, Task 5) and the status state is defined and not busy. The refreshed node has no `zarrbridge_state`, so this fires once. While `unindexed` it also fires once, so the node drops its pending flag and the `FileViewer` gate is the only thing keeping the hex dump away.
  - **Transition** (covers `building` to `multiscale`, where `zarr.json` gains levels): the previous state, kept in a `useRef`, was busy and the new one is not.
  - The refresh is `fileQuery.refetch()` and, when it resolves, `queryClient.invalidateQueries({ queryKey: ['zarr', 'metadata', fspName, path] })` with `useQueryClient()` from TanStack Query. Never call `zarrMetadataQuery.refetch()`: with `files` still empty it would run the query function and cache `metadata: null` as fresh for five minutes, and because the query key is path-only, enabling it later would not refetch. Invalidation only marks the entry stale, so a disabled query stays untouched and an enabled one refetches with the refreshed `files`.
- Produces: `export function shouldRefreshListing(prevState: string | undefined, state: string | undefined, nodeState: string | undefined): boolean` in `zarrQueries.ts`: `state !== undefined && !isZarrbridgeBusy(state) && (nodeState !== undefined || (prevState !== undefined && isZarrbridgeBusy(prevState)))`.

- [ ] **Step 1: Write the failing test**

Add `frontend/src/__tests__/unitTests/zarrbridgeRefresh.test.ts`:

```ts
import { shouldRefreshListing } from '@/queries/zarrQueries';

describe('shouldRefreshListing', () => {
  it('refreshes a pending node as soon as the status is terminal, even on the first answer', () => {
    expect(shouldRefreshListing(undefined, 'native', 'indexing')).toBe(true);
    expect(shouldRefreshListing(undefined, 'unindexed', 'indexing')).toBe(true);
  });
  it('waits while the status is still busy or unknown', () => {
    expect(shouldRefreshListing(undefined, 'indexing', 'indexing')).toBe(false);
    expect(shouldRefreshListing('indexing', undefined, 'indexing')).toBe(false);
  });
  it('refreshes on a busy to terminal transition of a complete node', () => {
    expect(shouldRefreshListing('building', 'multiscale', undefined)).toBe(true);
    expect(shouldRefreshListing('native', 'native', undefined)).toBe(false);
  });
});
```

Run: `pixi run test-frontend -- src/__tests__/unitTests/zarrbridgeRefresh.test.ts`
Expected: FAIL, `shouldRefreshListing` is not exported

- [ ] **Step 2: Implement `shouldRefreshListing` in `zarrQueries.ts`, then make the `FileBrowser` changes above**

- [ ] **Step 3: Run the test, type-check and lint**

Run: `pixi run node-check && pixi run node-eslint-write && pixi run node-prettier-write`
Expected: no new errors

- [ ] **Step 4: Verify in the running app**

With the dev server running (`pixi run --no-install uvicorn fileglancer.server:app --host 0.0.0.0 --port 7878 --reload --ssl-keyfile /opt/certs/cert.key --ssl-certfile /opt/certs/cert.crt` so the editable zarrbridge is used), open `https://nextflow.int.janelia.org:7878/browse/nrs_dudman/classy_midbrain/EASIFISH/Gel2/stitching/g2_r0/Gel2_075x_GFP_Tpbg_633_Ribo550_nDapi.czi`.
Expected: a spinner with "Reading image structure" for about 60 s, no hex dump, then the warning "Too large to index automatically" with the `zarrbridge index` command, and polling stops (no more `_status` lines in the server log). Open a small TIFF: no spinner flash, entries appear on first load. Confirm in `top` that the dataserver's reads stop within a few seconds of the timeout.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ui/BrowsePage/FileBrowser.tsx frontend/src/queries/zarrQueries.ts frontend/src/__tests__/unitTests/zarrbridgeRefresh.test.ts
git commit -m "feat: show zarrbridge progress for image files before their listing exists"
```

---

## Not in this plan

- The Fileglancer App that runs `zarrbridge index` as a cluster job (the request defers it). The `unindexed` reason already names the command the App will run.
- Making the build lock NFS-safe across hosts (`flock` with `local_lock=all` mounts); separate change.
- A refkv "trust the directory" mode that would make CZI opens O(directory); upstream spec change.
- A zarrbridge release and pin bump in fileglancer so deployments without the editable install get this; do after both branches land.
