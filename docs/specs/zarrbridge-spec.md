# zarrbridge: virtual OME-Zarr serving for legacy image files

**Status:** draft for implementation
**Working name:** `zarrbridge` (PyPI name free as of 2026-11-03; rename freely)
**Repo:** new, `JaneliaSciComp/zarrbridge`, BSD-3
**Relationship to existing code:** generalises `JaneliaSciComp/mrc-ng-server` (its serving rules and builder are ported in); builds on `d-v-b/refkv` for indexing and OME-Zarr conventions; borrows the thin-server shape of `psobolewskiPhD/refkv-server`. Does not use chunkmirage, VirtualiZarr, bffile (v1), or Neuroglancer precomputed output.

---

## 1. Summary

zarrbridge serves image files that are not Zarr (MRC, TIFF in its many forms, and the formats refkv parses: CZI, ND2, LIF, NDPI, SVS/QPTIFF, IMS, DICOM, NIfTI, NRRD, FITS) as **OME-Zarr 0.5 over HTTP**, without converting or duplicating the files. It is a library plus a small ASGI server plus a CLI.

The design has two halves with a hard line between them:

- **The request path does O(one chunk) of work.** A request for a chunk is a seek and a read of the file's own storage unit (a strip, a tile, a run of rows), optionally decoded and re-encoded when the viewer cannot decode the native codec. Nothing is re-chunked, resampled or computed on request. Data is served at its **native chunking**; for plane-chunked formats that means XY slice viewing, and the server says so rather than pretending otherwise.
- **Everything else is a separate process over a separate budget.** Multiscale levels above the native resolution are produced by a streaming builder that runs once per file, in the background or as an explicit job, writes a sidecar cache, and records a fingerprint. The server serves those levels only while the fingerprint validates, and advertises single resolution otherwise. The server never builds.

Those are mrc-ng-server's rules ("never compute on the request path", "cached info is the authority", "fingerprint or nothing"), applied to any reader instead of one.

It runs standalone as a personal data server (`zarrbridge serve /path/to/data`, like x2s3), and it exposes the contracts (library API, unix-socket + trusted-header mode, status endpoint, builder CLI) that Fileglancer needs to run it as a per-user data worker behind data links. The Fileglancer side of that integration is a separate spec.

### Goals

- Any viewer or library that reads OME-Zarr 0.5 over plain HTTP (Neuroglancer, vizarr, napari-ome-zarr, zarr-python, zarrita) can open a supported file with no plugins and no custom codecs.
- Zero copies of level 0. The only bytes written are multiscale levels (about ⅓ of the original for `2_2_1`, ⅐ for `2_2_2`) and small index/fingerprint files.
- Bounded memory per process regardless of file size, on both the request path and in the builder.
- Correctness under change: a replaced or edited file never serves stale pixels with a 200.
- Deployable by one person on a laptop against a directory, and by Fileglancer as an unprivileged per-user process.

### Non-goals (v1)

- Re-chunking (e.g. strips → cubes), computed levels on request, any image operation other than block-mean downsampling in the builder.
- Neuroglancer precomputed, N5, or zarr v2 output. (zarr v2 **input** is out of scope too; tensorstore and zarr-python already read it.)
- Bio-Formats/bffile reader (planned v1.1 behind an optional extra; the reader interface is designed for it).
- Authentication beyond a shared token for the standalone server. Multi-user auth is Fileglancer's job.
- Writing a hybrid vzip (references + pyramid in one archive). The cache is plain zarr arrays beside the data so every tool can read it and eviction is a `rm -r`.

---

## 2. Terms

- **Dataset**: one source file (or one refkv-recognised multi-file set) plus its derived state. Addressed by a filesystem path under a configured root.
- **Reader**: turns a source file into a zarr v3 `Store` presenting an OME-Zarr 0.5 hierarchy at native chunking. Three in v1: `mrc`, `tiff`, `refkv`.
- **Index**: for refkv-backed readers, the `ReferenceStore` describing where every chunk's bytes are. Persisted as a `.vzip`.
- **Pyramid cache** (or just **cache**): the directory holding levels ≥ 1, `info.json` and `fingerprint.json`.
- **Fingerprint**: the record that binds a cache to the exact source bytes and the exact code version that produced it.
- **Store stack**: the ordered zarr `Store` wrappers between a reader and the HTTP layer.

---

## 3. Design principles (normative)

1. **Request path = one chunk.** A chunk GET may: read that chunk's bytes from the source (one or a bounded number of `pread`s), decode it with the native codec, re-encode with the output codec, and return. It may not read other chunks, other levels, or run anything whose cost scales with the image.
2. **Native chunking, with one exception.** The chunk grid presented at level 0 is the file's storage grid (pages/strips/tiles; refkv's runs for flat files). Exception: a reader may present a viewer-friendly grid when the source is **uncompressed and seekable**, because then any box is O(box) reads with no decode. The `mrc` reader uses this (64³ cubes, as mrc-ng-server does in production); a future `raw`/uncompressed-TIFF path may too. Compressed slabs are never re-chunked.
3. **Cached info is the authority.** When a valid cache exists, the group `zarr.json` is served verbatim from `cache/info.json`; levels and chunks come from the same build and cannot disagree. Without a cache, the server generates single-resolution metadata from the reader.
4. **Fingerprint or nothing.** A cache is used only if its fingerprint matches the source identity and the current code versions. Any mismatch → native-only, and the status explains why.
5. **The server never writes the cache.** Builds are separate processes (spawned by the server in auto mode, or run by a CLI/job). A server-side chunk miss on a cached level is a 404, never a build.
6. **Unprivileged, relocatable, no global state.** The library runs as whatever user launched it, writes only under the cache locations it was given and `$XDG_CACHE_HOME/zarrbridge`, and needs no network except to read remote sources when configured.
7. **Honest classification.** If a file cannot be served well (compressed whole-plane chunks over a size threshold, a format refkv rejects and tifffile cannot read), the status says `unservable` with a reason, and the UI can route to conversion. Serving badly is a bug.

---

## 4. Architecture

```
                 ┌────────────── zarrbridge process (unprivileged) ──────────────┐
HTTP ──► ASGI app ─► DatasetResolver ─► Dataset ─► StoreStack ─► zarr-http-server │
            │             (longest file   │           │                            │
            │              prefix, LRU)   │           ├─ MetadataStore   (info.json authority)
            │                             │           ├─ PyramidOverlay  (mount cache/1..N)
            ├─ /_status, /_health, /      │           ├─ TranscodeStore  (codec policy)
            │                             │           └─ ReaderStore     (mrc | tiff | refkv)
            │                             ├─ Fingerprint (validate cache)
            └─ AutoBuildQueue ──spawn──►  Builder (subprocess)  ──writes──► cache dir
```

Packages:

```
zarrbridge/
  readers/   base.py, mrc/, tiff/, refkv_reader.py, registry.py
  store/     transcode.py, overlay.py, metadata.py, keys.py, codecs.py
  cache/     layout.py, fingerprint.py, locks.py, status.py, prune.py
  build/     builder.py, downsample.py, plan.py, queue.py
  server/    app.py, resolver.py, headers.py, browse.py, config.py
  api.py     (public library surface, §12)
  cli.py
```

---

## 5. Readers

### 5.1 Interface

```python
class Reader(Protocol):
    name: str                    # "mrc" | "tiff" | "refkv"
    version: int                 # bumped when output changes; part of the fingerprint

    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool: ...
    def open(self, path: Path, options: ReaderOptions) -> Opened: ...

@dataclass
class Opened:
    store: zarr.abc.store.Store  # read-only, zarr v3, OME-Zarr 0.5 hierarchy at native chunking
    images: list[ImageInfo]      # one per multiscales group: group path, shape, dtype, axes, units, scale, native chunk grid, codecs
    identity: SourceIdentity     # size, mtime_ns, header_hash, reader-specific extras (e.g. refkv pins)
    classification: Classification  # servable | servable_2d | unservable(reason)
    close: Callable[[], None]
```

Readers never write. Readers must be safe to open many times in one process (an LRU of `Opened` objects is kept per dataset) and must bound their own open file handles (an `FdCache` ported from mrc-ng-server is provided for readers that use raw fds).

### 5.2 `mrc` reader (port of mrc-ng-server)

- Port `mrcheader.py` (modes 0/1/2/6/12, axis maps, `cella`, `origin`, `nsymbt`, `exttyp`), `reader.py` (row-wise vs span-wise `pread`, threshold, strategy reporting) and `server/fdcache.py`.
- Chunk mode `cubes` (default): chunk shape `(64,64,64)` configurable; a chunk GET performs the ported read plan. Chunk mode `native`: delegate to the `refkv` reader (1 MiB runs) for xarray-style access.
- Metadata: axes `z y x` (+ `c`/`t` for stacks per the `tilt`/`stack` policy), units `angstrom` from `cella/mx` with the OME `scale` in the file's unit; `omero.window` from `dmin/dmax`; the full header mirrored under `attributes["mrc"]`.
- Stack policy (ported): files matched by configured glob (`*.st`, `*.ali`, `*_tilt.mrc`, …) are image stacks: no z downsampling (`2_2_1`), per-plane independence. Classification by configuration, never by shape inference.
- `identity.header_hash` = sha256 of the 1024-byte header + extended header.

### 5.3 `tiff` reader (tifffile)

- Accepts everything tifffile opens: plain multi-page, ImageJ hyperstacks (`imagej_metadata` gives axes), OME-TIFF (all images, via `series`), pyramidal SubIFDs (as stored levels), striped and tiled, any codec tifffile decodes.
- Native grid: tile shape for tiled pages, `(rows_per_strip, width)` for strips, one page per leading index. Chunks are produced by decoding exactly one segment (`TiffPage.segments()` → `decode`) and encoding with the output codec. Short last strips are padded to the chunk grid by the reader.
- Stored pyramids (SubIFDs, OME-TIFF resolutions) are presented as levels with scale factors computed from shapes; `multiscales.datasets[i].coordinateTransformations` use the `(f−1)/2` pixel-centre translation.
- `identity.header_hash` = sha256 of the first 64 KiB plus the IFD offset table bytes (so appended pages change identity).
- Index persistence: v1 re-parses on open. (v1.1: persist the segment offset/bytecount arrays as `.npz` in the cache dir keyed by fingerprint, because walking 100k IFDs over NFS is minutes.)

### 5.4 `refkv` reader

- `store = refkv.zarrify(path, policy=..., connections=...)` → `ReferenceStore`; `opened.store = refkv.zarr.store(store)`. Index persisted with `store.save_vzip(cache/index.vzip)` and reopened with `refkv.open(...)` when the fingerprint validates; pins from the store are copied into `identity.extras["pins"]` and re-verified against `stat` on open.
- Covers CZI, ND2, LIF, NDPI, SVS, QPTIFF, IMS, DICOM, NIfTI, NRRD, FITS, OME-TIFF and TIFF that refkv accepts. refkv's own OME-Zarr 0.5 conventions (axes, units, scale/translation, `omero`, source metadata mirrors, multi-image hierarchies) are used as emitted; zarrbridge adds nothing to level-0 metadata.
- Chunk values are references; zarr-python resolves them through the refkv store with refkv's `max_reads`/`max_concurrency` policy. Custom codecs refkv registers (`refkv.tiff_lzw`, `refkv.stream`, `packbits`, imagecodecs) are available to the transcode layer in-process; they are never exposed to clients (§6.2).
- Dependency pinning: refkv is pre-1.0 and its spec changes weekly. Pin an exact version (git tag or commit) in `pyproject.toml`; record `refkv.__version__` in the fingerprint; CI runs against the pin only. The surface used is deliberately small: `zarrify`, `ReferenceStore.save_vzip`, `refkv.open`, `refkv.zarr.store`, pins, and `refkv.compose.{mount,overlay}` (optional, §6.3).

### 5.5 Reader selection

`registry.select(path)`:

1. `.mrc/.rec/.st/.ali/.map` + MRC magic → `mrc`.
2. TIFF magic → `refkv` if `zarrify` accepts; else `tiff`. Configurable `readers.tiff.prefer = refkv | tifffile`. (Default `refkv` for zero-decode serving of raw/zstd data, pins and index caching; `tifffile` is the fallback for ImageJ stacks without OME-XML, Deflate/zstd strips with a short tail, old-style JPEG, and anything else refkv rejects.)
3. Anything else refkv `sniff`s → `refkv`.
4. Else `unservable: no reader`.

The chosen reader and the reason are part of `/_status`.

---

## 6. Store stack

Every layer is a read-only zarr v3 `Store`. Keys are OME-Zarr 0.5 paths relative to the dataset root (`zarr.json`, `<image>/zarr.json`, `<image>/0/zarr.json`, `<image>/0/c/0/3/7`, …).

### 6.1 `ReaderStore`

The `Opened.store` from the reader. For `mrc` and `tiff` it is a custom `Store` subclass that materialises chunks on `get`; for `refkv` it is `refkv.zarr.store(...)`.

### 6.2 `TranscodeStore` (codec policy)

- Allowlist of client-decodable codecs: `bytes`, `gzip`, `zstd`, `blosc`, `crc32c`, `transpose`, `sharding_indexed`. Configurable.
- For any array whose codec chain contains a codec outside the allowlist, this layer rewrites the array `zarr.json` to `[bytes(endian=little), zstd(level=3)]` (configurable) and, on chunk `get`, decodes through zarr-python's codec pipeline for the original chain and re-encodes. Arrays already on the allowlist pass through byte-for-byte (a refkv reference to a raw or gzip strip is sent as stored).
- Decode concurrency is bounded per process (`limits.max_decodes`, default `os.cpu_count()`); excess requests wait.

### 6.3 `PyramidOverlay`

When `Fingerprint.validate(cache)` is `valid`: keys `<image>/<k>/…` for `k ≥ 1` are served from the cache directory (a local zarr v3 directory store); everything else falls through. Implementation: a small overlay store (prefix routing) in zarrbridge; `refkv.compose.mount` may be used when both sides are `ReferenceStore`s, but a plain Python overlay is sufficient and avoids depending on compose semantics.

### 6.4 `MetadataStore` (cached info authority)

- Valid cache: `<image>/zarr.json` is served verbatim from `cache/<image>/info.json`. The builder wrote it, listing level 0 (native) and every built level with correct `scale`/`translation`.
- No valid cache: `<image>/zarr.json` is generated from the reader with only the native level(s) (plus stored pyramids the reader found). Marked with `attributes["zarrbridge"]["levels"] = "native"` and the status reason.
- Metadata responses carry `Cache-Control: no-cache` and an `ETag` (§8.3) so viewers pick up new levels after a build on reload without stale intermediaries.

---

## 7. Cache, fingerprint, validity

### 7.1 Location

```
cache.mode = beside | central         # default: beside if the file's directory is writable by this process, else central
cache.central_root = ~/.cache/zarrbridge/pyramids   # or an operator path
```

`beside`: `<dir>/.zarrbridge/<filename>/`. Permissions follow the data; a pyramid one lab member builds is readable by anyone who can read the file. `central`: `<central_root>/<sha256(abs_path)[:2]>/<sha256(abs_path)>/`, with the absolute path recorded inside for `prune` and `status`.

### 7.2 Layout

```
<cache>/
  fingerprint.json      # written LAST, after fsync of everything else; its presence means "complete"
  build.json            # progress/state while building and the final report
  lock                  # host:pid:started_at, flock'd while building
  index.vzip            # refkv index (refkv reader only)
  <image>/info.json     # the group zarr.json to serve verbatim
  <image>/1/zarr.json, <image>/1/c/...   # zarr v3 arrays, zstd(3), chunk shape per §9.3
  <image>/2/...
```

Multiple images in one file (OME-TIFF series, LIF, CZI scenes) share one cache dir with one fingerprint.

### 7.3 Fingerprint (port of `mrcng/fingerprint.py`, generalised)

```json
{
  "schema_version": 1,
  "derivation_version": 1,
  "zarrbridge_version": "0.3.0",
  "reader": {"name": "tiff", "version": 2},
  "refkv_version": "0.0.0+gabcdef0",
  "source": {"path": "/abs/path/file.tif", "size": 123, "mtime_ns": 456, "header_hash": "sha256:...", "pins": [...]},
  "plan": {"factor": [2,2,1], "levels": 4, "level_chunks": [[1,512,512], ...], "codec": "zstd:3", "method": "mean"},
  "built_at": "2026-11-03T12:00:00Z", "built_by": "alice@host", "duration_s": 812
}
```

Versions and what bumps them:

- `schema_version`: the shape of this JSON. Mismatch → `incompatible`.
- `derivation_version`: the meaning of the chunks beside it (downsampling method, padding, translation convention, chunk-key encoding). Bumped by hand in `zarrbridge/cache/fingerprint.py` with a changelog line. Mismatch → `outdated`.
- `reader.version`: per-reader output changes (axis order, metadata). Mismatch → `outdated`.
- `source.*`: identity. Mismatch → `stale`.

`validate(cache_dir, opened) -> valid | stale | outdated | incompatible | missing | building`. Only `valid` serves levels. Every other state is reported in `/_status` with the field that mismatched.

Validation is cheap (one `stat`, one small file read, header hash compared from the reader's already-parsed identity) and is performed on dataset open and re-checked at most every `cache.recheck_seconds` (default 30) while a dataset is held open, by `fstat` on the reader's fd rather than `stat` on the path, so NFS attribute caching cannot hide a replaced file behind the same path.

---

## 8. Server

### 8.1 Routing

```
GET /_health                         → 200 {"ok": true, "datasets_open": n, "builds": {...}}
GET /                                → browse page (optional, --browse)
GET /_status/{root}/{path}           → status JSON (§8.4)
GET /{root}/{path}/{zarr-key}        → store key
```

`{root}` is a configured root name (several roots, like x2s3 targets). `{path}` is resolved by the longest prefix under the root that is a regular file (as refkv-server does), with path containment and symlink policy (`--follow-symlinks` off by default). The remainder is the zarr key. Datasets are held open in an LRU (`limits.max_open_datasets`, default 64) with idle close.

Keys are served by zarr-http-server's `store_app` mounted per dataset (one app per open dataset, created lazily), wrapped by zarrbridge middleware for headers, logging and limits. If zarr-http-server's API proves too narrow (no header hooks today), vendor its ~300 lines rather than fork; contribute hooks upstream in parallel.

### 8.2 Trusted-path mode (for Fileglancer)

`--trusted-path-header X-Zarrbridge-Path` with `--uds /run/.../zb.sock` (or `--trusted-proxy 127.0.0.1`): the absolute source path comes from the header set by the reverse proxy (percent-encoded UTF-8, since header values are ASCII; a plain ASCII path needs no encoding), the URL path is the zarr key only, and `{root}/{path}` resolution is skipped. Requests without the header are 400. The header value is still subjected to containment against `--allow-prefix` globs. This is the contract that lets Fileglancer resolve a sharing key to a path in its own process and hand zarrbridge only the result.

### 8.3 HTTP semantics

- `ETag`: `W/"<fingerprint-hash>:<key>"` for cached levels; `W/"<identity-hash>:<key>"` for native level and metadata. HEAD and `If-None-Match` → 304 supported.
- `Cache-Control`: metadata `no-cache`; native-level chunks `public, max-age=3600, must-revalidate`; cached-level chunks `public, max-age=86400, must-revalidate`. `--immutable-chunks` switches chunk responses to `max-age=31536000, immutable` for deployments whose links carry the fingerprint in the URL (Fileglancer can).
- Byte ranges on chunk GETs (zarr-python sharded reads) pass through zarr-http-server.
- CORS: `--cors-origin` repeatable; default none (standalone personal use typically sets `*`).
- `X-Zarrbridge-Source: native|cache|transcoded` and `X-Zarrbridge-Read: rowwise|spanwise|reference` on chunk responses, for benchmarks (port of mrc-ng's strategy header).
- Standalone auth: `--token` requires `Authorization: Bearer` or `?token=`; binding to a non-loopback host without `--token` requires `--insecure`.

### 8.4 Status JSON

```json
{
  "path": "/abs/file.tif", "reader": "tiff", "reader_reason": "refkv rejected: ImageJ stack without OME-XML",
  "classification": "servable_2d", "images": [{"path": "0", "shape": [1000,4096,4096], "dtype": "uint16", "native_chunks": [1,128,4096]}],
  "levels": {"state": "building", "built": 2, "planned": 5, "progress": 0.41, "eta_s": 300, "reason": null},
  "cache": {"mode": "beside", "dir": "/data/.zarrbridge/file.tif", "bytes": 1234567, "fingerprint": "valid|stale|...", "mismatch": null},
  "advice": ["xy_only"]
}
```

`levels.state ∈ {native, indexing, queued, building, multiscale, failed, unservable}`. Viewers and Fileglancer read this to show the three-state badge and to decide whether to offer "Build multiscale".

### 8.5 Auto-build

```
build.auto = on | off               # default on for standalone, off under trusted-path mode unless Fileglancer enables it
build.auto_max_bytes = 20 GiB       # above this, never automatic; status says "queued: run `zarrbridge build` or submit a job"
build.slots = 1                     # host-level concurrent builder subprocesses for this server
build.nice = 10; build.ionice = idle
```

On first open of a dataset whose cache is `missing|stale|outdated|incompatible` and whose size is under `auto_max_bytes`, the server enqueues a build. The queue spawns `zarrbridge build <path>` as a subprocess (never in-process), honouring `slots`; the builder takes the lock, so concurrent servers or CLIs on the same file coordinate through the filesystem. Dedupe by cache dir.

### 8.6 Limits

`limits.max_open_datasets`, `limits.chunk_cache_bytes` (small LRU of encoded chunks, default 256 MiB), `limits.max_decodes`, `limits.max_reads_per_chunk` (refuse chunks whose plan exceeds it; pairs with classification), `limits.max_chunk_bytes` (chunks above it make the dataset `unservable` unless uncompressed, default 64 MiB).

### 8.7 Logging

JSON access log per request: `path, key, status, bytes, duration_ms, source (native|cache|transcoded), read_strategy, cache_hit, bytes_read_from_source`. One line per dataset open/close with reader choice and classification. Builder events (§9.6) at INFO.

---

## 9. Builder

### 9.1 Invocation

`zarrbridge build <file|dir> [--factor 2,2,1|2,2,2|auto] [--levels N] [--min-axis 64] [--chunk ...] [--codec zstd:3] [--jobs J] [--shard i/n] [--force] [--cache-mode beside|central] [--report out.json]`

Exit codes: `0` built, `3` `SKIPPED_LOCKED` (another builder holds the lock), `4` `SKIPPED_VALID` (fingerprint already valid, no `--force`), `5` `UNSERVABLE`, `1` error.

### 9.2 Downsample factor policy (`auto`)

- `2_2_1` when: level-0 chunks are single planes (`chunk[z] == 1`), or the reader marks the dataset as a stack (MRC stack policy, ImageJ `frames`/`channels` without z spacing), or z spacing > 2× xy spacing.
- `2_2_2` when: 3D chunks and isotropic within 10 % (the `mrc` cubes case).
- Otherwise `2_2_1`. Always overridable per invocation and per glob in config (`build.factor_by_glob`).
- Stop when every downsampled axis would fall below `--min-axis` (default 64) or at `--levels` (default 8). Channels and time are never downsampled.

### 9.3 Level layout

- Level `k` chunk shape = level-0 chunk shape, clamped to the level's shape, except that a plane-slab chunk `(1, R, W)` becomes `(1, min(R', 512), min(W', 512))`-style tiles when `W/2^k < W` would leave full-width slabs that are too thin; concretely: tiles of `512×512` in the downsampled axes for 2D-chunked data, cubes of the native edge for 3D-chunked data. The plan is recorded in the fingerprint.
- Codec: `zstd(level=3)`; dtype preserved; block mean for intensity data with rounding; for integer label data (configurable `build.method = mean | mode`) mode is v1.1.
- `coordinateTransformations`: `scale = scale0 * factor^k`, `translation = scale0 * (factor^k − 1)/2` per downsampled axis (pixel-centre convention, identical to mrc-ng-server's `omezarr.py`).
- `info.json` is written after all levels, listing level 0 as served by the reader (its metadata copied from the reader's generated `zarr.json`) and all built levels.

### 9.4 Streaming algorithm (port of `mrcng/pyramid.py`, generalised)

- Read level 0 **in the source's native chunk order** through the same `ReaderStore` the server uses (so every reader gets the builder for free), decoding each native chunk exactly once. For `mrc` cubes mode the builder reads full-width rows instead (the port of mrc-ng's 16× fewer-bytes trick).
- Level 1 is produced in row blocks: hold one output row block and the input chunks that cover it; emit, release. Level `k+1` is produced from level `k`'s on-disk chunks (cascade), so memory never exceeds a few input chunks plus one output block.
- Writes go to `<level>.partial/` and are renamed into place per level; `fingerprint.json` last after `fsync`. A crash leaves no half-valid cache.
- `--jobs J` parallelises by plane ranges within one process (threads for I/O-bound readers, processes for `tiff` decode). `--shard i/n` lets n cluster jobs each build a z-range of level 1 into `1.partial/shard-i/` with a final `zarrbridge build --merge` step; v1 implements `--jobs`, reserves `--shard` (v1.1).

### 9.5 Locking and progress

`lock` holds `host:pid:started_at` under `flock`. A lock older than `build.lock_stale_seconds` (default 6 h) whose pid is not alive on the same host is broken with a logged warning. `build.json` is updated every N chunks with `{state, level, done, total, started_at, eta_s}` and finalised with the report (`bytes_read`, `bytes_written`, `duration_s`, `levels`).

### 9.6 Status/prune CLI

- `zarrbridge status <file|dir|root>`: table of datasets with reader, classification, levels state, cache size, fingerprint state and mismatch field.
- `zarrbridge prune <dir|central_root> [--older-than 90d] [--max-bytes N] [--stale]`: delete caches whose fingerprint is `stale|outdated|incompatible`, orphaned central caches whose source path no longer exists, then LRU by `atime` of `fingerprint.json` until under `--max-bytes`. Never touches a locked cache.
- `zarrbridge index <file>`: write the refkv index only. `zarrbridge inspect <file>`: print what would be served (reader, grid, codecs, transcode decision, planned pyramid size and estimated build time at a given MB/s), without serving or building.

---

## 10. Configuration

TOML file (`zarrbridge.toml`), environment (`ZB_*`), CLI flags; precedence CLI > env > file.

```toml
[server]
host = "127.0.0.1"; port = 8800; uds = ""; token = ""; cors_origins = []; browse = true
trusted_path_header = ""; allow_prefixes = ["/groups/**", "/nrs/**"]; immutable_chunks = false

[roots]
data = "/path/to/data"
scratch = "/path/to/scratch"

[cache]
mode = "beside"; central_root = "~/.cache/zarrbridge/pyramids"; recheck_seconds = 30

[build]
auto = true; auto_max_bytes = "20GiB"; slots = 1; nice = 10
factor = "auto"; min_axis = 64; max_levels = 8; codec = "zstd:3"; method = "mean"
[build.factor_by_glob]
"*.st" = "2,2,1"

[readers.mrc]
chunk_mode = "cubes"; chunk_shape = [64,64,64]; stack_globs = ["*.st", "*.ali", "*_tilt.mrc"]
[readers.tiff]
prefer = "refkv"
[readers.refkv]
connections = 8; max_reads = 64

[limits]
max_open_datasets = 64; chunk_cache_bytes = "256MiB"; max_decodes = 0; max_chunk_bytes = "64MiB"
```

---

## 11. Deployment modes

### 11.1 Standalone (personal data server)

`zarrbridge serve /path/to/data --port 8800 --cors-origin '*'` → browse page at `/` with Neuroglancer and vizarr links, auto-build on, caches beside the data. Single process; uvicorn; no root. A Dockerfile and a `systemd --user` unit example ship in `deploy/`.

### 11.2 Under Fileglancer (per-user data worker)

Contract this spec provides; the Fileglancer side (supervisor `kind: dataserver`, nginx `auth_request` resolve, link type, status badge, "Build multiscale" App) is the next spec.

- Process: `zarrbridge serve --uds <path> --trusted-path-header X-Fg-Resolved-Path --allow-prefix ... --build-auto <policy> --cache-mode beside --no-browse --log-format json`, launched as the data owner with no root, no network listener, config via env `ZB_*`.
- Builder: `zarrbridge build <file> --report <json>` runnable by a Fileglancer App on the cluster as the user, writing the same cache layout; `--shard` for large files (v1.1).
- Status: `/_status/` (trusted-path mode: `GET /_status` with the header) is the single source for the UI badge.
- Headers: zarrbridge emits `ETag`/`Cache-Control` as in §8.3 so nginx's `proxy_cache` can absorb fan-out; Fileglancer may append `?v=<fingerprint>` to links and enable `--immutable-chunks`.
- Footprint budget per process (so the supervisor can plan): idle ≤ 150 MiB; `limits.*` make the ceiling configurable; no JVM in v1.

---

## 12. Library API (`zarrbridge.api`)

```python
ds = zarrbridge.open(path, config=Config(...))        # -> Dataset (reader selected, cache validated)
ds.store                                               # the composed zarr v3 Store (use with zarr.open_group(ds.store))
ds.status()                                            # the §8.4 dict
ds.ensure_pyramid(mode="never" | "auto" | "now")      # never: no-op; auto: enqueue if eligible; now: run the builder in-process (CLI/jobs only)
zarrbridge.build(path, plan=..., report=...)           # what the CLI calls
zarrbridge.create_app(config) -> ASGI                  # what `serve` runs; embeddable
zarrbridge.Fingerprint.validate(cache_dir, opened)     # exposed for tooling
```

Everything the server does is reachable from this API; the server adds only HTTP.

---

## 13. Dependencies

Python ≥ 3.12 (refkv's floor). `refkv` (exact pin), `zarr>=3`, `numpy`, `numcodecs`/`zstandard`, `tifffile`, `starlette`, `uvicorn`, `zarr-http-server` (or vendored), `tomllib` (stdlib). Optional extras: `imagecodecs` (JPEG/JPEG 2000/JPEG XR transcoding), `bffile` (v1.1), `obstore` if remote roots are enabled.

---

## 14. Testing

- **Fixtures** (`tests/fixtures/`, generated by a script, small): MRC modes 0/1/2/6 incl. a tilt stack; TIFF uncompressed strips, LZW strips with a short tail, Deflate strips, 512² tiles, ImageJ hyperstack (TZCYX), OME-TIFF 3-series with SubIFD pyramid, BigTIFF; plus refkv's own fixtures for CZI/ND2/LIF/NDPI where redistributable.
- **Parity with mrc-ng-server**: for each MRC fixture, byte-identical level-0 chunks in `cubes` mode, byte-identical level ≥1 chunks given the same factor, and semantically identical OME-Zarr metadata (documented, enumerated diffs only: group attribute namespace, `zarrbridge` attribute block).
- **Conformance**: every served `zarr.json` validates against the NGFF 0.5 JSON schema; every image round-trips through `zarr.open_group` + `ome-zarr-py` reader; `zarrita` (node) opens the standalone server in CI.
- **Request-path invariant**: a chunk GET reads ≤ `limits.max_reads_per_chunk` ranges and ≤ `2 × chunk_bytes + segment_bytes` from the source (instrumented reader; asserted per fixture). A GET on a missing cached level is 404 and spawns nothing.
- **Fingerprint state machine**: touch/replace/append the source → `stale`; bump `derivation_version` → `outdated`; bump `schema_version` → `incompatible`; concurrent `build` → one builds, one `SKIPPED_LOCKED`; crash mid-build (kill −9) → `missing`, no partial serving.
- **Builder memory**: build a 4 GiB synthetic stack under `RLIMIT_AS` of 1 GiB.
- **Transcode**: LZW/JPEG fixtures served as `zstd` with pixel-identical content (lossless) or identical decoded pixels (lossy) versus tifffile direct reads.
- **Trusted-path mode**: header required, containment enforced, URL path is the key only.
- **Benchmarks** (not CI): port `mrcng/benchmark.py`; p50/p95/p99 per `read_strategy` and per `source`; a slice-scroll and an XZ-view scenario against the TIFF stack.

---

## 15. Milestones

1. **M1 Core serving**: reader interface, `mrc` (cubes) and `tiff` readers, `TranscodeStore`, `MetadataStore` (native only), resolver, server with headers/status/health/browse, config, `serve` + `inspect`. Neuroglancer opens an MRC and a TIFF stack.
2. **M2 Cache**: fingerprint, cache layout, builder (streaming, cascade, locks, progress), `PyramidOverlay`, `build/status/prune`, auto-build queue, parity tests against mrc-ng-server, benchmark port.
3. **M3 refkv**: `refkv` reader, index persistence, pins, reader selection with fallback, multi-image hierarchies, conformance suite, CZI/ND2/LIF/NDPI fixtures.
4. **M4 Integration surface**: trusted-path + UDS mode, library API freeze, JSON logging, Dockerfile and user-unit, docs, 0.1 release. (Fileglancer integration spec starts here.)
5. **v1.1**: tifffile index persistence, `--shard`, label-aware downsampling, bffile reader behind an extra, resume builds.

---

## 16. Risks and open questions

1. **refkv churn.** Mitigated by the exact pin and the small surface; the risk is a needed fix landing only on a newer, incompatible spec. Decide early whether Janelia will carry a fork tag.
2. **tifffile decode throughput** for LZW stacks viewed in XZ (thousands of strip decodes per screen). `limits.max_decodes`, the chunk LRU and the `xy_only` advice are the mitigations; conversion is the answer past a threshold. Measure in M1.
3. **`beside` cache on group directories** with unusual permissions (setgid, ACLs, read-only mounts). The `central` fallback must be automatic and visible in status.
4. **Default MRC chunk mode.** `cubes` keeps production parity with mrc-ng-server; it violates "native chunking" by design under the uncompressed-and-seekable exception. Confirm this is the intended default for tomograms (proposed: yes).
5. **Browse page scope** in standalone mode: directory listing with viewer links only (as mrc-ng/refkv-server), no thumbnails in v1.
6. **Name.** `zarrbridge` is a placeholder; decide before the first release so the cache directory name (`.zarrbridge/`) is stable.
