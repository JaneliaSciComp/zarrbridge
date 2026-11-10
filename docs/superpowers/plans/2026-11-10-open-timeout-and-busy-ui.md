# Bounded opens and a busy UI for zarrbridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** zarrbridge gives up indexing a file after 60 s instead of working for an hour in the background, reports what it is doing while it works, says clearly when a file is too big to index automatically and that it must be indexed manually, and Fileglancer shows a working animation whenever zarrbridge is busy on the file being viewed.

**Architecture:** The refkv zarrify is run as a cancellable coroutine with a deadline; a timeout raises a new `IndexTimeout`. The dataset resolver opens files on a background thread, dedupes concurrent opens, remembers timeouts for a short while, and lets `_status` answer without blocking, so the status endpoint reports `indexing` during the open and the new terminal state `unindexed` after a timeout. Fileglancer's `virtual_node` consults `_status` first and returns a pending root node, its catch-all stops turning transport errors into "plain file", and the frontend drives everything off `levels.state`: a spinner for the three transient states, warning copy for `unindexed`, and a listing refetch when the state leaves a transient value.

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
- The `_status` key is reachable from inside the hierarchy (`image.tif/0/_status`) and the pending answer must be the same there. Pinned by Task 3's `test_status_pending_shape` which calls both forms.

---

### Task 1: A cancellable, time-bounded zarrify

**Files:**
- Modify: `zarrbridge/_sync.py`
- Modify: `zarrbridge/readers/base.py`
- Modify: `zarrbridge/server/config.py:105-116` (`LimitsConfig`)
- Modify: `zarrbridge/readers/refkv_reader.py:84-92` (`_zarrify`)
- Test: `tests/test_refkv_reader.py`

**Interfaces:**
- Produces: `run_sync_with_timeout(coro: Awaitable[T], timeout: float | None) -> T` in `zarrbridge/_sync.py`; raises builtin `TimeoutError` after cancelling the coroutine's task on the background loop.
- Produces: `class IndexTimeout(ReaderRejected)` in `zarrbridge/readers/base.py` with `__init__(self, path: Path, timeout_s: float)` and attribute `timeout_s`; its message is `f"indexing did not finish within {timeout_s:.0f} s, so this file is too big to index automatically. It needs to be indexed manually: run `zarrbridge index {path}`; the saved index makes later opens fast."`.
- Produces: `LimitsConfig.open_timeout_s: float = 60.0`, `LimitsConfig.open_grace_s: float = 2.0`, `LimitsConfig.open_retry_s: float = 60.0`, `LimitsConfig.index_save_after_s: float = 5.0`. Zero disables each.
- Produces: `_zarrify(path: Path, config) -> ReferenceStore` raising `IndexTimeout`.

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
    cfg.limits.open_timeout_s = 0.2
    t0 = time.time()
    with pytest.raises(IndexTimeout, match="too big to index automatically"):
        _zarrify(tmp_path / "x.czi", cfg)
    assert time.time() - t0 < 2
    time.sleep(0.1)
    assert state["cancelled"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e test test tests/test_refkv_reader.py::test_zarrify_timeout_cancels_and_raises`
Expected: FAIL with `ImportError: cannot import name 'IndexTimeout'`

- [ ] **Step 3: Implement `run_sync_with_timeout` in `zarrbridge/_sync.py`**

Schedule with `asyncio.run_coroutine_threadsafe(coro, _bg_loop())`, call `.result(timeout)`; on `concurrent.futures.TimeoutError` call `fut.cancel()` and raise `TimeoutError`. Always use the background loop, even when no loop runs in the caller's thread, so cancellation has a loop to run on.

- [ ] **Step 4: Add `IndexTimeout` to `readers/base.py` and the four limits to `LimitsConfig`**

- [ ] **Step 5: Rewrite `_zarrify` to use `refkv.zarrify.zarrify_async` through `run_sync_with_timeout`**

Timeout is `config.limits.open_timeout_s or None`. Translate `TimeoutError` into `IndexTimeout(path, timeout_s)`. Import `zarrify_async` from the module at call time (`refkv.zarrify.zarrify_async`) so the test's monkeypatch is seen.

- [ ] **Step 6: Run the reader tests**

Run: `pixi run -e test test tests/test_refkv_reader.py`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add zarrbridge/_sync.py zarrbridge/readers/base.py zarrbridge/server/config.py zarrbridge/readers/refkv_reader.py tests/test_refkv_reader.py
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
- Produces: `DatasetResolver.dataset(path: Path, wait: float | None = None) -> Dataset`. Starts an open if none is in flight, joins the in-flight one otherwise, waits up to `wait` seconds (None: until the open ends, which the Task 1 deadline bounds), then returns the dataset, or raises `StillOpening`, `IndexTimeout`, or `ReaderRejected`.
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

- [ ] **Step 4: Implement `cached` and `open_state`**

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
- Consumes: `resolver.dataset(path, wait=config.limits.open_grace_s)`, `resolver.open_state(path)`, `StillOpening`, `IndexTimeout` from Tasks 1 and 2.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e test test tests/test_server.py -k "pending_shape or fast_file or unindexed"`
Expected: FAIL (`pending_shape` blocks and asserts on elapsed time; `unindexed` gets `unservable`)

- [ ] **Step 3: Rewrite the `status` route**

Call `resolver.dataset(path, wait=config.limits.open_grace_s)`. On `StillOpening` build the pending JSON (`elapsed_s = time.time() - e.started_at`, cache fields from `cache_dir_for(path, config)` and `cache_writable`). On `IndexTimeout` build the unindexed JSON. Keep the `ReaderRejected` branch as it is. Note `IndexTimeout` is a `ReaderRejected`, so catch it first.

- [ ] **Step 4: Update `key_handler`**

`resolver.dataset(path)` with no wait; catch `IndexTimeout` before `ReaderRejected` and return `PlainTextResponse(f"unindexed: {e}", status_code=422)`.

- [ ] **Step 5: Update the spec**

§8.4: add `unindexed` to the state set with one sentence: "`unindexed` is terminal: the open hit `limits.open_timeout_s`; the file is too big to index automatically and must be indexed manually with `zarrbridge index`, after which the saved index makes the next open fast." Add: "`_status` never blocks for longer than `limits.open_grace_s`; while a dataset is being opened it reports `indexing` with `elapsed_s`." §8.6: add `limits.open_timeout_s` (60 s; the zarrify is cancelled, no work continues), `open_grace_s` (2 s), `open_retry_s` (60 s; a timed-out file is not reopened sooner), `index_save_after_s` (5 s; see Task 4).

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pixi/envs/test/bin/python -m pytest tests/test_zarrproxy.py -k "while_indexing or transport_error"`
Expected: FAIL (`seen` includes `/zarr.json`; the second returns 200 with the plain file)

- [ ] **Step 3: Implement `PENDING_STATES` and the root pre-check in `virtual_node`**

- [ ] **Step 4: Narrow the catch-all in `_virtual_node`**

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

**Interfaces:**
- Consumes: `isZarrbridgeBusy`, `useZarrbridgeStatusQuery` from Task 6; `FileOrFolder.virtual_zarr` and `zarrbridge_state` from Task 5.
- Produces: `isVirtualZarr = fileQuery.data?.currentFileOrFolder?.virtual_zarr === true` (no longer requires `isZarrDir`). The status query is enabled by `isVirtualZarr` alone.
- Produces: the hint renders when `isVirtualZarr && (zarrbridgeStatusQuery.data || zarrbridgeStatusQuery.isError)`, passing `state` (or `'error'` with `reason = error.message` on a failed query) and `progress`.
- Produces: `ZarrPreview` renders only when `isZarrDir && zarrMetadataQuery.data?.metadata`; `FileViewer` renders only when the item is a file, `!isZarrDir` and `!virtual_zarr`; when `isVirtualZarr && !isZarrDir` the main area below the hint is empty.
- Produces: a `useEffect` keyed on `zarrbridgeStatusQuery.data?.levels?.state` that calls `fileQuery.refetch()` and `zarrMetadataQuery.refetch()` when the previous state was busy and the new one is not (track the previous value in a `useRef`).

- [ ] **Step 1: Make the changes**

- [ ] **Step 2: Type-check and lint**

Run: `pixi run node-check && pixi run node-eslint-write && pixi run node-prettier-write`
Expected: no new errors

- [ ] **Step 3: Verify in the running app**

With the dev server running (`pixi run --no-install uvicorn fileglancer.server:app --host 0.0.0.0 --port 7878 --reload --ssl-keyfile /opt/certs/cert.key --ssl-certfile /opt/certs/cert.crt` so the editable zarrbridge is used), open `https://nextflow.int.janelia.org:7878/browse/nrs_dudman/classy_midbrain/EASIFISH/Gel2/stitching/g2_r0/Gel2_075x_GFP_Tpbg_633_Ribo550_nDapi.czi`.
Expected: a spinner with "Reading image structure" for about 60 s, no hex dump, then the warning "Too large to index automatically" with the `zarrbridge index` command, and polling stops (no more `_status` lines in the server log). Open a small TIFF: no spinner flash, entries appear on first load. Confirm in `top` that the dataserver's reads stop within a few seconds of the timeout.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ui/BrowsePage/FileBrowser.tsx
git commit -m "feat: show zarrbridge progress for image files before their listing exists"
```

---

## Not in this plan

- The Fileglancer App that runs `zarrbridge index` as a cluster job (the request defers it). The `unindexed` reason already names the command the App will run.
- Making the build lock NFS-safe across hosts (`flock` with `local_lock=all` mounts); separate change.
- A refkv "trust the directory" mode that would make CZI opens O(directory); upstream spec change.
- A zarrbridge release and pin bump in fileglancer so deployments without the editable install get this; do after both branches land.
