"""ASGI app (spec §8). The store stack does the work; this layer adds routing, headers, auth and logging."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype

from zarrbridge import __version__
from zarrbridge.build.queue import AutoBuildQueue
from zarrbridge.readers.base import ReaderRejected
from zarrbridge.server import browse
from zarrbridge.server.config import Config
from zarrbridge.server.headers import cache_control, content_type, read_strategy
from zarrbridge.server.resolver import DatasetResolver, Forbidden, NotFound

_access = logging.getLogger("zarrbridge.access")
_log = logging.getLogger("zarrbridge.server")
EXPOSE = ["Content-Range", "ETag", "X-Zarrbridge-Source", "X-Zarrbridge-Read", "Content-Length"]


def create_app(config: Config) -> Starlette:
    trusted = bool(config.server.trusted_path_header)
    if not trusted and not config.roots:
        raise ValueError("no roots configured (use `serve <dir>` or [roots] in zarrbridge.toml)")
    if config.server.host not in ("127.0.0.1", "localhost", "::1", "") and not config.server.uds \
            and not config.server.token and not config.server.insecure:
        raise ValueError("binding a non-loopback host without --token requires --insecure")
    queue = AutoBuildQueue(config) if config.build.auto else None
    resolver = DatasetResolver(config, queue)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        resolver.close_all()
        if queue:
            queue.shutdown()

    app = Starlette(lifespan=lifespan, routes=_routes(config, resolver, queue, trusted),
                    middleware=_middleware(config))
    app.state.config, app.state.resolver, app.state.queue = config, resolver, queue
    return app


class ViewerProxy:
    """GET passthrough for a hosted Neuroglancer (fileglancer's dev approach). Buffers each response; assets
    with a content hash in the name are cached for the process lifetime, index.html for `ttl` seconds."""

    FORWARD = ("content-type", "cache-control", "etag", "last-modified")

    def __init__(self, upstream: str, ttl: float = 300.0) -> None:
        self.upstream = upstream.rstrip("/")
        self.ttl = ttl
        self._cache: dict[str, tuple[float, int, bytes, dict[str, str]]] = {}

    def fetch(self, path: str) -> tuple[int, bytes, dict[str, str]]:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(f"{self.upstream}/{path}", headers={"Accept-Encoding": "identity",
                                                                          "User-Agent": "zarrbridge"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read(), {k: v for k, v in r.headers.items() if k.lower() in self.FORWARD}
        except urllib.error.HTTPError as e:
            return e.code, b"", {}

    async def handle(self, request: Request) -> Response:
        path = request.path_params.get("path", "") or "index.html"
        if any(seg in ("..", ".") for seg in path.split("/")) or any(c in path for c in "?#%"):
            return Response(status_code=400)
        hashed = bool(re.search(r"\.[0-9a-f]{8,}\.", path))
        hit = self._cache.get(path)
        if hit and (hashed or time.time() - hit[0] < self.ttl):
            _, status, body, headers = hit
        else:
            try:
                status, body, headers = await asyncio.to_thread(self.fetch, path)
            except OSError as e:
                return PlainTextResponse(f"could not reach {self.upstream}: {e}", status_code=502)
            if status == 200:
                self._cache[path] = (time.time(), status, body, headers)
        if hashed:
            headers = dict(headers, **{"Cache-Control": "public, max-age=31536000, immutable"})
        return Response(body, status_code=status, headers=headers)


def viewer_static_dir(config: Config) -> Path | None:
    """A standalone Neuroglancer client build to serve at /neuroglancer/ (same origin: no CORS, no Chrome
    Local Network Access permission). The pip `neuroglancer` package's client is not standalone."""
    d = Path(os.path.expanduser(config.server.viewer_dir)) if config.server.viewer_dir else None
    return d if d and (d / "index.html").exists() else None


def has_viewer(config: Config) -> bool:
    """Whether /neuroglancer/ is served on this origin (a build on disk or a passthrough to viewer_url)."""
    return bool(config.server.trusted_path_header) is False and (viewer_static_dir(config) is not None or bool(config.server.viewer_url))


def _middleware(config: Config) -> list[Middleware]:
    mw: list[Middleware] = []
    if config.server.cors_origins:
        # allow_private_network: a public page (neuroglancer-demo, vizarr) fetching from a private address sends a
        # Private Network Access preflight; without this Chrome blocks the request outright.
        mw.append(Middleware(CORSMiddleware, allow_origins=config.server.cors_origins, allow_methods=["GET", "HEAD"],
                             allow_private_network=True,
                             allow_headers=["Range", "If-None-Match", "Authorization", config.server.trusted_path_header or "X-Zarrbridge-Path"],
                             expose_headers=EXPOSE, max_age=3600))
    return mw


def _routes(config: Config, resolver: DatasetResolver, queue: AutoBuildQueue | None, trusted: bool) -> list[Route]:
    token = config.server.token

    def authorized(request: Request) -> bool:
        if not token:
            return True
        auth = request.headers.get("authorization", "")
        return auth == f"Bearer {token}" or request.query_params.get("token") == token

    async def health(request: Request) -> Response:
        return JSONResponse({"ok": True, "version": __version__, "datasets_open": resolver.open_count,
                             "builds": queue.snapshot() if queue else None, "mode": "trusted-path" if trusted else "roots"})

    def guard(handler):
        async def wrapped(request: Request) -> Response:
            if not authorized(request):
                return PlainTextResponse("unauthorized", status_code=401)
            return await handler(request)
        return wrapped

    def locate(request: Request) -> tuple[Path, str]:
        """(file path, zarr key) for this request, in either mode."""
        if trusted:
            hdr = request.headers.get(config.server.trusted_path_header)
            if not hdr:
                raise _Http(400, f"missing {config.server.trusted_path_header} header")
            return resolver.check_trusted(hdr), request.path_params.get("key", "").strip("/")
        return resolver.split(request.path_params["root"], request.path_params.get("path", ""))

    async def status(request: Request) -> Response:
        try:
            path, _ = locate(request)
            ds = resolver.dataset(path)
        except _Http as e:
            return JSONResponse({"error": e.detail}, status_code=e.status)
        except Forbidden as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        except NotFound as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except ReaderRejected as e:
            return JSONResponse({"path": str(path), "classification": "unservable", "reader": None,
                                 "reader_reason": str(e), "levels": {"state": "unservable", "reason": str(e)}}, status_code=200)
        ds.recheck()
        return JSONResponse(ds.status(queue))

    async def key_handler(request: Request) -> Response:
        t0 = time.perf_counter()
        key = ""
        try:
            path, key = locate(request)
        except _Http as e:
            return PlainTextResponse(e.detail, status_code=e.status)
        except Forbidden as e:
            return PlainTextResponse(str(e), status_code=403)
        except NotFound:
            if not trusted and config.server.browse:
                return _browse(request, config, resolver)
            return Response(status_code=404)
        if not key:
            if config.server.browse and not trusted:
                return _redirect_to_listing(request, path)
            return Response(status_code=404)
        segs = key.split("/")
        if any(s in ("", ".", "..") for s in segs) or "\x00" in key:
            return Response(status_code=404)
        try:
            ds = resolver.dataset(path)
        except ReaderRejected as e:
            return PlainTextResponse(f"unservable: {e}", status_code=422)
        ds.recheck()
        resp = await _serve_key(request, ds, key, config)
        _access.info(json.dumps({"path": str(path), "key": key, "status": resp.status_code,
                                 "bytes": int(resp.headers.get("content-length") or 0),
                                 "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                                 "source": resp.headers.get("x-zarrbridge-source"),
                                 "read_strategy": resp.headers.get("x-zarrbridge-read")}))
        return resp

    async def root(request: Request) -> Response:
        if config.server.index_html:
            try:
                return HTMLResponse(Path(config.server.index_html).read_text())
            except OSError:
                return Response(status_code=404)
        if trusted or not config.server.browse:
            return Response(status_code=404)
        return HTMLResponse(browse.render_roots(config.roots))

    routes = [Route("/_health", health)]
    viewer_dir = viewer_static_dir(config) if not trusted else None
    if viewer_dir is not None:
        from starlette.staticfiles import StaticFiles
        routes.append(Mount("/neuroglancer", app=StaticFiles(directory=str(viewer_dir), html=True), name="neuroglancer"))
    elif has_viewer(config):
        from starlette.responses import RedirectResponse
        proxy = ViewerProxy(config.server.viewer_url)
        routes.append(Route("/neuroglancer", lambda r: RedirectResponse("/neuroglancer/")))  # NG asset URLs are relative
        routes.append(Route("/neuroglancer/{path:path}", proxy.handle))
    if trusted:
        routes += [Route("/_status", guard(status)),
                   Route("/{key:path}", guard(key_handler), methods=["GET", "HEAD"])]
    else:
        routes += [Route("/", guard(root)),
                   Route("/_status/{root}/{path:path}", guard(status)),
                   Route("/{root}/{path:path}", guard(key_handler), methods=["GET", "HEAD"]),
                   Route("/{root}", guard(key_handler), methods=["GET", "HEAD"])]
    return routes


class _Http(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status, self.detail = status, detail


def _parse_range(header: str | None):
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].strip()
    if "," in spec:
        return None
    start, _, end = spec.partition("-")
    try:
        if start and end:
            return RangeByteRequest(int(start), int(end) + 1)
        if start:
            return OffsetByteRequest(int(start))
        if end:
            return SuffixByteRequest(int(end))
    except ValueError:
        return None
    return None


async def _serve_key(request: Request, ds, key: str, config: Config) -> Response:
    etag = ds.etag_for(key)
    headers = {"ETag": etag, "Cache-Control": cache_control(ds, key, config.server.immutable_chunks),
               "X-Zarrbridge-Source": ds.source_of(key), "Accept-Ranges": "bytes"}
    rs = read_strategy(ds, key)
    if rs:
        headers["X-Zarrbridge-Read"] = rs
    inm = request.headers.get("if-none-match")
    if inm and etag in [t.strip() for t in inm.split(",")]:
        return Response(status_code=304, headers=headers)
    byte_range = _parse_range(request.headers.get("range")) if request.method == "GET" else None
    proto = default_buffer_prototype()
    try:
        buf = await ds.store.get(key, proto, byte_range)
    except FileNotFoundError:
        buf = None
    if buf is None:
        if byte_range is not None and await ds.store.get(key, proto) is not None:
            return Response(status_code=416, headers=headers)
        return Response(status_code=404, headers={"Cache-Control": "no-cache"})
    body = buf.to_bytes()
    headers["Content-Type"] = content_type(key)
    if request.method == "HEAD":
        headers["Content-Length"] = str(len(body))
        return Response(status_code=200, headers=headers)
    if byte_range is not None:
        full = await ds.store.getsize(key) if not isinstance(byte_range, RangeByteRequest) else None
        start = byte_range.start if isinstance(byte_range, RangeByteRequest) else (
            byte_range.offset if isinstance(byte_range, OffsetByteRequest) else (full or 0) - len(body))
        total = full if full is not None else "*"
        headers["Content-Range"] = f"bytes {start}-{start + len(body) - 1}/{total}"
        return Response(body, status_code=206, headers=headers)
    return Response(body, status_code=200, headers=headers)


def _browse(request: Request, config: Config, resolver: DatasetResolver) -> Response:
    root = request.path_params.get("root", "")
    rel = request.path_params.get("path", "").strip("/")
    try:
        base = resolver.root_dir(root)
    except NotFound:
        return Response(status_code=404)
    target = (base / rel) if rel else base
    if any(s in (".", "..") for s in rel.split("/")) or not resolver._contained(target, base) or not target.is_dir():
        return Response(status_code=404)
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    viewer = f"{base_url}/neuroglancer/" if has_viewer(config) else None
    return HTMLResponse(browse.render_listing(root, rel, target, base_url, viewer))


def _redirect_to_listing(request: Request, path: Path) -> Response:
    from starlette.responses import RedirectResponse
    url = str(request.url.path).rstrip("/")
    return RedirectResponse(url.rsplit("/", 1)[0] + "/", status_code=302)
