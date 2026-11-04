"""zarrbridge CLI: serve | build | status | prune | index | inspect."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from zarrbridge import __version__
from zarrbridge.server.config import Config, parse_bytes


def _parse_duration(s: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
    s = s.strip()
    if s[-1:] in units:
        return float(s[:-1]) * units[s[-1]]
    return float(s)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zarrbridge", description="Virtual OME-Zarr 0.5 serving for legacy image files")
    ap.add_argument("--version", action="version", version=f"zarrbridge {__version__}")
    ap.add_argument("--config", type=Path, default=os.environ.get("ZB_CONFIG"), help="zarrbridge.toml")
    ap.add_argument("--cache-mode", choices=["beside", "central"])
    ap.add_argument("--central-root")
    ap.add_argument("--log-format", choices=["text", "json"])
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve one or more directories")
    s.add_argument("paths", nargs="*", type=Path, help="directories to serve (root name = basename)")
    s.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    s.add_argument("--host"); s.add_argument("--port", type=int); s.add_argument("--uds")
    s.add_argument("--token"); s.add_argument("--insecure", action="store_true")
    s.add_argument("--cors-origin", action="append", default=None)
    s.add_argument("--browse", dest="browse", action="store_true", default=None)
    s.add_argument("--no-browse", dest="browse", action="store_false")
    s.add_argument("--trusted-path-header"); s.add_argument("--allow-prefix", action="append", default=None)
    s.add_argument("--trusted-proxy", help="accepted for compatibility; use a reverse proxy that strips the header")
    s.add_argument("--build-auto", choices=["on", "off"])
    s.add_argument("--immutable-chunks", action="store_true", default=None)
    s.add_argument("--follow-symlinks", action="store_true", default=None)
    s.add_argument("--ssl-certfile"); s.add_argument("--ssl-keyfile")
    s.add_argument("--viewer-dir", help="standalone Neuroglancer client build (dist/client) to serve at /neuroglancer/")
    s.add_argument("--viewer-url", help="hosted Neuroglancer to pass through at /neuroglancer/ (default neuroglancer-demo.appspot.com; '' disables)")

    b = sub.add_parser("build", help="build the multiscale cache for a file or every servable file in a dir")
    b.add_argument("target", type=Path)
    b.add_argument("--factor", help="auto | 2,2,1 | 2,2,2 (x,y,z)")
    b.add_argument("--levels", type=int); b.add_argument("--min-axis", type=int)
    b.add_argument("--chunk", help="reserved"); b.add_argument("--codec", default=None)
    b.add_argument("--jobs", type=int, default=1); b.add_argument("--shard", help="reserved (v1.1)")
    b.add_argument("--force", action="store_true"); b.add_argument("--report", type=Path)

    st = sub.add_parser("status", help="table of datasets and their cache state")
    st.add_argument("target", type=Path)
    st.add_argument("--json", action="store_true")

    pr = sub.add_parser("prune", help="delete invalid/orphaned/old caches")
    pr.add_argument("target", type=Path)
    pr.add_argument("--older-than"); pr.add_argument("--max-bytes"); pr.add_argument("--stale", action="store_true")
    pr.add_argument("--dry-run", action="store_true")

    ix = sub.add_parser("index", help="write the refkv index only")
    ix.add_argument("target", type=Path)

    ins = sub.add_parser("inspect", help="print what would be served, without serving or building")
    ins.add_argument("target", type=Path)
    ins.add_argument("--mbps", type=float, default=200.0, help="assumed read throughput for the ETA")
    ins.add_argument("--json", action="store_true")
    return ap


def _config(args: argparse.Namespace) -> Config:
    ov: dict[str, Any] = {"server": {}, "cache": {}, "build": {}}
    if args.cache_mode:
        ov["cache"]["mode"] = args.cache_mode
    if args.central_root:
        ov["cache"]["central_root"] = args.central_root
    if args.log_format:
        ov["server"]["log_format"] = args.log_format
    if args.cmd == "serve":
        roots = {}
        for p in args.paths:
            roots[Path(p).resolve().name or "root"] = str(Path(p).resolve())
        for r in args.root:
            name, _, path = r.partition("=")
            roots[name] = str(Path(path).resolve())
        if roots:
            ov["roots"] = roots
        for k in ("host", "port", "uds", "token", "insecure", "browse", "trusted_path_header", "immutable_chunks", "follow_symlinks",
                  "ssl_certfile", "ssl_keyfile", "viewer_dir"):
            v = getattr(args, k)
            if v is not None and v is not False or (k == "insecure" and v):
                ov["server"][k] = v
        if args.viewer_url is not None:
            ov["server"]["viewer_url"] = args.viewer_url
        if args.cors_origin is not None:
            ov["server"]["cors_origins"] = args.cors_origin
        if args.allow_prefix is not None:
            ov["server"]["allow_prefixes"] = args.allow_prefix
        if args.build_auto:
            ov["build"]["auto"] = args.build_auto == "on"
        elif args.trusted_path_header:
            ov["build"]["auto"] = False  # default off under trusted-path mode (spec §8.5)
    if args.cmd == "build" and args.codec:
        ov["build"]["codec"] = args.codec
    return Config.load(file=args.config, overrides=ov)


def _logging(config: Config, verbose: int) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING
    fmt = '{"time":"%(asctime)s","logger":"%(name)s","level":"%(levelname)s","msg":%(message)s}' \
        if config.server.log_format == "json" else "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    logging.getLogger("zarrbridge.access").setLevel(logging.INFO if config.server.log_format == "json" else logging.WARNING)
    logging.getLogger("zarrbridge.build").setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)
    _logging(config, args.verbose)
    return {"serve": _serve, "build": _build, "status": _status, "prune": _prune, "index": _index,
            "inspect": _inspect}[args.cmd](args, config)


def _serve(args, config: Config) -> int:
    import uvicorn
    from zarrbridge.server.app import create_app
    app = create_app(config)
    kw: dict[str, Any] = {"log_level": "info" if args.verbose else "warning", "access_log": False}
    if config.server.uds:
        kw["uds"] = config.server.uds
    else:
        kw.update(host=config.server.host, port=config.server.port)
    if config.server.ssl_certfile:
        kw.update(ssl_certfile=config.server.ssl_certfile, ssl_keyfile=config.server.ssl_keyfile or None)
    scheme = "https" if config.server.ssl_certfile else "http"
    print(f"zarrbridge {__version__} serving roots {list(config.roots) or '(trusted-path mode)'} on "
          f"{config.server.uds or f'{scheme}://{config.server.host}:{config.server.port}/'}", file=sys.stderr)
    uvicorn.run(app, **kw)
    return 0


def _build(args, config: Config) -> int:
    from zarrbridge.build.builder import build
    from zarrbridge.cache.status import iter_files
    target = Path(args.target)
    files = list(iter_files(target))
    if not files:
        print(f"{target}: no servable files", file=sys.stderr)
        return 5
    worst = 0
    for i, f in enumerate(files):
        rep = args.report if (args.report and len(files) == 1) else (
            Path(f"{args.report}.{i}.json") if args.report else None)
        rc = build(f, config, factor=args.factor, levels=args.levels, min_axis=args.min_axis, force=args.force,
                   jobs=args.jobs, report=rep)
        print(json.dumps({"path": str(f), "exit": rc, "status": {0: "built", 3: "skipped_locked", 4: "skipped_valid",
                                                                   5: "unservable", 1: "error"}.get(rc, "error")}))
        if rc not in (0, 4) and worst == 0:
            worst = rc
    return worst


def _status(args, config: Config) -> int:
    from zarrbridge.cache.status import dataset_status_row, iter_files
    rows = [dataset_status_row(f, config) for f in iter_files(Path(args.target))]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    cols = ["path", "reader", "classification", "levels", "cache_bytes", "fingerprint", "mismatch"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return 0


def _prune(args, config: Config) -> int:
    from zarrbridge.cache.prune import prune
    removed = prune(Path(args.target), config,
                    older_than_s=_parse_duration(args.older_than) if args.older_than else None,
                    max_bytes=parse_bytes(args.max_bytes) if args.max_bytes else None,
                    stale_only=args.stale, dry_run=args.dry_run)
    for d in removed:
        print(("would remove " if args.dry_run else "removed ") + str(d))
    return 0


def _index(args, config: Config) -> int:
    from zarrbridge.cache.layout import INDEX, cache_dir_for
    from zarrbridge.readers.base import ReaderRejected
    from zarrbridge.readers.refkv_reader import RefkvReader, save_index
    path = Path(args.target).resolve()
    cache_dir, _ = cache_dir_for(path, config)
    try:
        op = RefkvReader().open(path, config)
    except ReaderRejected as e:
        print(f"{path}: {e}", file=sys.stderr)
        return 5
    save_index(op, cache_dir / INDEX)
    print(str(cache_dir / INDEX))
    return 0


def _inspect(args, config: Config) -> int:
    import numpy as np
    from zarrbridge.build.plan import choose_factor, plan_levels
    from zarrbridge.dataset import Dataset
    from zarrbridge.readers.base import ReaderRejected
    path = Path(args.target).resolve()
    try:
        ds = Dataset.open(path, config)
    except ReaderRejected as e:
        print(json.dumps({"path": str(path), "classification": "unservable", "reason": str(e)}, indent=2))
        return 5
    try:
        st = ds.status()
        plans = []
        est_bytes = 0
        for img in ds.opened.images:
            f = choose_factor(img, config, path)
            lv = plan_levels(img, f, config.build.min_axis, config.build.max_levels)
            itemsize = np.dtype(img.dtype).itemsize if img.dtype else 2
            lvl_bytes = [int(np.prod(l.shape)) * itemsize for l in lv[1:]]
            est_bytes += sum(lvl_bytes) // 3  # zstd guess
            plans.append({"image": img.path, "factor": list(f), "levels": [{"path": str(l.index), "shape": list(l.shape),
                          "chunks": list(l.chunks), "scale": l.scale} for l in lv],
                          "transcoded": img.path in ds.opened.transcoded or img.level_path(0) in ds.opened.transcoded,
                          "codecs": [c.get("name") for c in img.codecs]})
        out = {**st, "plan": plans, "estimated_cache_bytes": est_bytes,
               "estimated_build_s": round(ds.opened.identity.size / (args.mbps * 1e6), 1)}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"{path}\n  reader: {st['reader']} ({st['reader_reason']})\n  classification: {st['classification']} {st['advice']}")
            for img in st["images"]:
                print(f"  image {img['path']!r}: shape {img['shape']} dtype {img['dtype']} axes {img['axes']} native chunks {img['native_chunks']} levels {img['levels']}")
            for pl in plans:
                print(f"  transcode: {'yes' if pl['transcoded'] else 'no'} (codecs {pl['codecs']}); factor {pl['factor']}")
                for l in pl["levels"]:
                    print(f"    level {l['path']}: shape {l['shape']} chunks {l['chunks']} scale {l['scale']}")
            print(f"  cache: {st['cache']['mode']} {st['cache']['dir']} fingerprint={st['cache']['fingerprint']} levels={st['levels']['state']}")
            print(f"  estimated cache bytes: {est_bytes:,}; estimated build time at {args.mbps:g} MB/s: {out['estimated_build_s']} s")
        return 0
    finally:
        ds.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
