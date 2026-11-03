# zarrbridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve MRC, TIFF and refkv-parsed image files as OME-Zarr 0.5 over HTTP at native chunking, with a separate-process pyramid builder guarded by a fingerprint.

**Architecture:** Three readers produce a read-only zarr v3 `Store` plus metadata; a store stack (Transcode → PyramidOverlay → Metadata) wraps it; a small Starlette app resolves `{root}/{path}/{key}` to a store `get` and adds headers. The builder streams level 0 through the same store into a sidecar cache and writes `fingerprint.json` last. The server never writes the cache.

**Tech Stack:** Python 3.12, zarr>=3.4, numpy, tifffile, refkv==0.2.0 (exact pin, PyPI), zstandard, starlette, uvicorn, tomllib. Test: pytest, httpx, mrcfile, ome-zarr-models.

**Spec:** `docs/specs/zarrbridge-spec.md`

## Global Constraints

- Python ≥ 3.12. `refkv==0.2.0` exact pin; `refkv.__version__` recorded in the fingerprint. refkv 0.2.0 has `ReferenceStore.save` (not `save_vzip`) and no `refkv.compose`: use `getattr(rs, "save_vzip", rs.save)`; overlay is a plain Python store.
- `zarr-http-server` is **not** used: its `store_app` has no header hooks and we need ETag/Cache-Control/X-Zarrbridge-* on every response. Our key handler is ~60 lines (§8.1 permits vendoring; this is smaller).
- Request path = one chunk. No store `get` may read other chunks or levels. Cached-level miss → 404, never build.
- Server never writes the cache. Only `zarrbridge build` (subprocess or CLI) writes under `<cache>/`.
- `fingerprint.json` is written last, after fsync of everything else; its absence means `missing`.
- Cache dir: `beside` → `<dir>/.zarrbridge/<filename>/`; `central` → `<central_root>/<sha256(abs)[:2]>/<sha256(abs)>/`.
- Output codec for built levels and transcoded chunks: `[bytes(endian=little), zstd(level=3)]`.
- Translation convention: `scale0 * (f^k − 1)/2` per downsampled axis; level 0 has no translation entry.
- Units: MRC `angstrom` (spec §5.2; differs from mrc-ng-server's nanometer, documented parity diff). Header mirror under `attributes["mrc"]`.
- Fingerprint versions start at `schema_version=1`, `derivation_version=1`, reader versions `mrc=1, tiff=1, refkv=1`.
- Package layout per spec §4 (readers/, store/, cache/, build/, server/, api.py, cli.py). Merge tiny modules; do not add more.
- Trusted-path mode: requests without the header are 400; header value is checked against `allow_prefixes` globs.
- Status `levels.state ∈ {native, indexing, queued, building, multiscale, failed, unservable}`.

## Review Focus

1. **Short last strip / edge tiles** in TIFF: the chunk must be zero-padded to the full chunk shape (zarr requires full chunks); test pins last-row chunk byte length == prod(chunk_shape)*itemsize. (Task 4)
2. **Big-endian TIFF**: uncompressed pass-through must declare `bytes(endian=big)`; compressed must decode then emit little-endian. Test with `byteorder=">"`. (Task 4)
3. **Replaced file under the same path while held open**: `Dataset.recheck()` uses `fstat` on the reader fd and flips the cache to `stale` within `recheck_seconds`; test replaces the file and asserts the next metadata GET has no level 1 and the status says `stale`. (Task 9)
4. **Path traversal / symlink escape** in the resolver: `..`, encoded slashes, and a symlink pointing outside the root must 404 unless `follow_symlinks`. (Task 10)
5. **Concurrent builders**: second `build` on a locked cache exits 3 and leaves the first's `.partial` untouched; a kill −9 mid-build leaves `missing`, and the server serves native only. (Task 8)

---

### Task 1: Package skeleton and configuration

**Files:**
- Create: `pyproject.toml`, `README.md`, `LICENSE` (BSD-3), `.gitignore`, `zarrbridge/__init__.py`, `zarrbridge/server/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass tree in `zarrbridge/server/config.py` with sections `server, roots, cache, build, readers, limits` mirroring spec §10 exactly (field names and defaults as in the TOML block). `Config.load(file: Path | None = None, env: Mapping[str, str] = os.environ, overrides: dict[str, Any] = {}) -> Config`. Precedence: overrides > env > file > defaults. Env keys: `ZB_<SECTION>_<FIELD>` uppercased (`ZB_SERVER_PORT`, `ZB_CACHE_MODE`, `ZB_ROOTS` as `name=path,name=path`). Lists from env are comma-separated. Byte strings (`"20GiB"`, `"256MiB"`) parsed by `parse_bytes(s: str | int) -> int`.
- `Config.to_dict()` for `/_status` and `inspect`.

- [ ] **Step 1: Write the failing test**

```python
def test_defaults():
    c = Config.load(file=None, env={})
    assert c.server.port == 8800 and c.cache.mode == "beside" and c.build.auto is True
    assert c.build.auto_max_bytes == 20 * 2**30 and c.limits.chunk_cache_bytes == 256 * 2**20
    assert c.readers.mrc.chunk_shape == (64, 64, 64) and c.readers.tiff.prefer == "refkv"

def test_precedence(tmp_path):
    f = tmp_path / "zb.toml"; f.write_text('[server]\nport = 1\n[roots]\ndata = "/d"\n')
    c = Config.load(file=f, env={"ZB_SERVER_PORT": "2"}, overrides={"server": {"port": 3}})
    assert c.server.port == 3 and c.roots == {"data": "/d"}
    assert Config.load(file=f, env={"ZB_SERVER_PORT": "2"}).server.port == 2

def test_parse_bytes():
    assert parse_bytes("20GiB") == 20 * 2**30 and parse_bytes("3MB") == 3_000_000 and parse_bytes(7) == 7
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_config.py -v` → ImportError.
- [ ] **Step 3: Implement** `pyproject.toml` (hatchling, `zarrbridge = "zarrbridge.cli:main"` script, deps per Global Constraints, extras `imagecodecs`, `test`), config dataclasses, `parse_bytes`.
- [ ] **Step 4: Run test to verify it passes.**
- [ ] **Step 5: Commit** `feat: package skeleton and config`.

---

### Task 2: Reader contracts, ChunkStore and codec helpers

**Files:**
- Create: `zarrbridge/readers/__init__.py`, `zarrbridge/readers/base.py`, `zarrbridge/store/__init__.py`, `zarrbridge/store/chunkstore.py`, `zarrbridge/store/codecs.py`, `zarrbridge/store/keys.py`
- Test: `tests/test_chunkstore.py`, `tests/test_codecs.py`

**Interfaces (produces):**
- `readers/base.py`:
  - `@dataclass(frozen=True) ImageInfo(path: str, shape: tuple[int,...], dtype: str, axes: list[dict], scale: list[float], native_chunks: tuple[int,...], codecs: list[dict], levels: list[str], is_stack: bool = False)` — `path` is the group path relative to dataset root (`""` for a single image at root); `levels` are dataset paths present natively (`["0"]`, or `["0","1"]` for stored pyramids).
  - `@dataclass(frozen=True) SourceIdentity(path: str, size: int, mtime_ns: int, header_hash: str, extras: dict = field(default_factory=dict))` with `.digest() -> str` = sha256 over `json.dumps({size, header_hash, extras}, sort_keys=True)`[:16] (mtime excluded so the ETag survives a `touch`; mtime still participates in fingerprint `stale`).
  - `@dataclass(frozen=True) Classification(state: Literal["servable","servable_2d","unservable"], reason: str | None = None)`.
  - `@dataclass Opened(store: Store, images: list[ImageInfo], identity: SourceIdentity, classification: Classification, reader_name: str, reader_version: int, reader_reason: str, close: Callable[[], None], fd: int | None = None, read_region: Callable[[str, tuple[slice,...]], np.ndarray] | None = None)`. `fd` is used for `fstat` rechecks; `read_region(image_path, slices)` is the optional builder fast path.
  - `class Reader(Protocol)`: `name: str; version: int; @classmethod sniff(cls, path: Path, head: bytes) -> bool; open(self, path: Path, config: Config) -> Opened`.
- `store/keys.py`: `split_chunk_key(key: str) -> tuple[str, tuple[int,...]] | None` (`"0/c/1/2/3"` → `("0", (1,2,3))`; `"img/0/c/4"` → `("img/0", (4,))`); `array_path_of(key) -> str | None`; `level_of(array_path) -> tuple[str, int] | None` (`"img/2"` → `("img", 2)`, `"0"` → `("", 0)`).
- `store/chunkstore.py`: `class ChunkStore(zarr.abc.store.Store)`: constructor `ChunkStore(metadata: dict[str, bytes], chunk_fn: Callable[[str, tuple[int,...]], bytes | None], grids: dict[str, tuple[int,...]])` where `grids[array_path]` = number of chunks per axis (for `list_prefix` and bounds). Read-only. `get` returns metadata bytes or calls `chunk_fn` via `asyncio.to_thread`, applying `byte_range`. `list`/`list_prefix`/`list_dir` enumerate metadata keys only (chunk keys are not listed; `supports_listing=True`). `getsize` for chunks = len(chunk_fn(...)) (acceptable: HEAD on a chunk costs one chunk).
- `store/codecs.py`: `OUTPUT_CODECS = [{"name":"bytes","configuration":{"endian":"little"}},{"name":"zstd","configuration":{"level":3,"checksum":False}}]`; `ALLOWLIST = {"bytes","gzip","zstd","blosc","crc32c","transpose","sharding_indexed","numcodecs.zlib"}`; `encode_chunk(arr: np.ndarray, codecs: list[dict] = OUTPUT_CODECS) -> bytes` (C-contiguous, little-endian cast, then zstd); `decode_chunk(data: bytes, array_meta: dict, chunk_shape: tuple[int,...]) -> np.ndarray` using zarr's `get_pipeline_class().from_codecs(ArrayV3Metadata.from_dict(meta).codecs)` and an `ArraySpec` (pattern verified against zarr 3.4.2); `pad_chunk(arr, chunk_shape) -> np.ndarray` (zeros, data in the corner; returns same object if already full); `codec_names(meta) -> list[str]`.

- [ ] **Step 1: Write failing tests**

```python
def test_split_chunk_key():
    assert split_chunk_key("0/c/1/2/3") == ("0", (1, 2, 3))
    assert split_chunk_key("img/0/c/4") == ("img/0", (4,))
    assert split_chunk_key("zarr.json") is None and level_of("img/2") == ("img", 2) and level_of("0") == ("", 0)

async def test_chunkstore_get_and_range():
    md = {"zarr.json": b"{}"}
    st = ChunkStore(md, lambda p, c: bytes(range(10)) if c == (0,) else None, {"0": (1,)})
    assert (await st.get("zarr.json", proto)).to_bytes() == b"{}"
    assert (await st.get("0/c/0", proto)).to_bytes() == bytes(range(10))
    assert (await st.get("0/c/0", proto, RangeByteRequest(2, 5))).to_bytes() == bytes([2, 3, 4])
    assert await st.get("0/c/1", proto) is None
    assert sorted([k async for k in st.list()]) == ["zarr.json"]

def test_encode_decode_roundtrip():
    a = np.arange(24, dtype=">u2").reshape(2, 3, 4)
    meta = {..."codecs": OUTPUT_CODECS, "chunk_grid": {"name":"regular","configuration":{"chunk_shape":[2,3,4]}}, "data_type":"uint16", "shape":[2,3,4] ...}
    assert np.array_equal(decode_chunk(encode_chunk(a), meta, (2,3,4)), a)
    assert pad_chunk(np.ones((1, 2), "u1"), (2, 3)).tolist() == [[1, 1, 0], [0, 0, 0]]
```

- [ ] **Step 2: Run** → fails. **Step 3: Implement.** **Step 4: Run** → passes. **Step 5: Commit** `feat: reader contracts, ChunkStore, codec helpers`.

---

### Task 3: mrc reader (port of mrc-ng-server)

**Files:**
- Create: `zarrbridge/readers/mrc/__init__.py`, `zarrbridge/readers/mrc/header.py`, `zarrbridge/readers/mrc/read.py`, `zarrbridge/readers/mrc/fdcache.py`, `zarrbridge/readers/mrc/reader.py`, `zarrbridge/store/omezarr.py`
- Test: `tests/conftest.py` (port `make_mrc` from mrc-ng-server `tests/conftest.py:11-61` verbatim, plus dmin/dmax at offset 76), `tests/test_mrc_reader.py`

**Interfaces:**
- Consumes: Task 2 types.
- Produces:
  - `header.py`: `MrcHeader` frozen dataclass = mrc-ng's fields (`nx ny nz mode nxstart nystart nzstart mx my mz nsymbt mapc mapr maps voxel_size_angstrom voxel_size_is_default is_image_stack mode0_signedness_is_ambiguous dtype data_offset file_size mtime_ns`) **plus** `dmin: float, dmax: float, dmean: float, cella: tuple[float,float,float], origin: tuple[float,float,float] (offset 196 "<3f"), exttyp: str (offset 104 "4s")`; properties `shape_zyx`, `served_dtype`. `parse_header(fd, file_size, mtime_ns, assume_mode0=None, is_image_stack=False) -> MrcHeader` with mrc-ng's exact rules (mode table, IMOD stamp, byte-order check, `(1,2,3)` axis order only, `STACK_Z_VOXEL_ANGSTROM=10.0`, voxel-size formula verbatim). `classify_path(relpath, stack_globs, volume_globs=()) -> bool`. Exceptions `MrcFormatError` + subclasses as in mrc-ng. `header_hash(fd, data_offset) -> str` = `"sha256:" + sha256(pread(fd, data_offset, 0))`.
  - `read.py`: `ReadStrategy`, `pread_exact`, `choose_strategy(x0, x1, itemsize, threshold=4096)`, `read_chunk(fd, hdr, x0, x1, y0, y1, z0, z1, row_bytes_threshold=4096) -> np.ndarray` verbatim from mrc-ng `reader.py:61-82` (clipped, served_dtype output).
  - `fdcache.py`: `FdCache(max_size=256)` port: `open(path) -> contextmanager[Handle(fd, hdr)]`, `close_all()`; validity memo dropped (fingerprint lives in Task 7).
  - `store/omezarr.py` (shared by all readers and the builder): `array_meta(shape, chunk_shape, dtype, codecs, dimension_names) -> dict`; `multiscales_group(images_axes, datasets, name, extra_attrs) -> dict`; `dataset_entry(path, scale, translation | None) -> dict`; `level_transforms(scale0: list[float], factor_k: list[int]) -> (scale, translation)` with `scale = s0*f`, `translation = s0*(f-1)/2`.
  - `reader.py`: `class MrcReader` (`name="mrc"`, `version=1`): `sniff` = suffix in `{.mrc,.rec,.st,.ali,.map}` and bytes 208:212 == `b"MAP "` or valid mode at offset 12; `open(path, config)`: cubes mode builds `ChunkStore` with keys `zarr.json` (group, axes `z y x` type space unit `angstrom`, scale = voxel_size_angstrom reversed to zyx, `omero.channels[0].window = {min: dmin, max: dmax, start: dmin, end: dmax}` when dmax > dmin, `attributes["mrc"]` = header as dict minus dtype/fd fields), `0/zarr.json` (array, chunk shape from `config.readers.mrc.chunk_shape`, codecs `[bytes little]`, dimension_names zyx, fill 0), chunk_fn reads via `read_chunk` and `pad_chunk`. `read_region(image, slices)` = `read_chunk` on the slice bounds. `Opened.identity = SourceIdentity(path, size, mtime_ns, header_hash, extras={"mode": mode, "is_image_stack": ...})`. Classification: `servable`; `unservable` with reason on `MrcFormatError`. `chunk_mode="native"` → raise `NotImplementedError("use refkv")` and the registry (Task 5) routes to refkv.

- [ ] **Step 1: Write failing tests**

```python
def test_header_parse_and_hash(tmp_path):
    p = make_mrc(tmp_path/"a.mrc", (80,60,40), mode=1, voxel_size_angstrom=6.8, dmin=0, dmax=29999)
    fd = os.open(p, os.O_RDONLY); st = os.stat(p)
    h = parse_header(fd, st.st_size, st.st_mtime_ns)
    assert h.shape_zyx == (40,60,80) and h.dtype == np.dtype("<i2") and h.voxel_size_angstrom == (6.8,6.8,6.8)
    assert header_hash(fd, h.data_offset) == "sha256:" + hashlib.sha256(open(p,"rb").read(1024)).hexdigest()

def test_cube_chunk_matches_mrcfile_padded(tmp_path):
    p = make_mrc(tmp_path/"a.mrc", (80,60,40), mode=1)
    op = MrcReader().open(p, Config.load(env={}))
    data = mrcfile.open(p, permissive=True).data           # zyx
    got = np.frombuffer(sync(op.store.get("0/c/0/0/1", proto)).to_bytes(), "<i2").reshape(64,64,64)
    exp = pad_chunk(data[0:40, 0:60, 64:80], (64,64,64))
    assert np.array_equal(got, exp)
    meta = json.loads(sync(op.store.get("zarr.json", proto)).to_bytes())
    ms = meta["attributes"]["ome"]["multiscales"][0]
    assert [a["unit"] for a in ms["axes"]] == ["angstrom"]*3 and ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0,1.0,1.0]
    assert meta["attributes"]["mrc"]["nx"] == 80 and meta["attributes"]["ome"]["omero"]["channels"][0]["window"]["max"] == 29999

def test_stack_glob_sets_z_voxel(tmp_path): ...  # *stack.mrc → is_stack, scale z == 10.0, images[0].is_stack

def test_zarr_opens_and_reads_back(tmp_path):
    g = zarr.open_group(op.store, mode="r"); assert np.array_equal(g["0"][:], data)
```

- [ ] **Step 2: Run** → fails. **Step 3: Implement** (port, do not redesign). **Step 4: Run** → passes. **Step 5: Commit** `feat: mrc reader (port of mrc-ng-server)`.

---

### Task 4: tiff reader (tifffile)

**Files:**
- Create: `zarrbridge/readers/tiff/__init__.py`, `zarrbridge/readers/tiff/reader.py`, `zarrbridge/readers/tiff/meta.py`
- Test: `tests/conftest.py` (TIFF fixture makers: `make_tiff(path, arr, **imwrite_kwargs)`, `make_ome_pyramid`, `make_imagej`), `tests/test_tiff_reader.py`

**Interfaces:**
- Produces: `class TiffReader` (`name="tiff"`, `version=1`): `sniff` = magic in `{b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"}`.
  - One `ImageInfo` per `TiffFile.series`; image path `""` if one series else `str(i)`; multi-series root `zarr.json` is a plain group with `attributes["zarrbridge"]["images"] = [paths]`.
  - Axes: map tifffile letters `T→t(time) Z→z(space) C→c(channel) Y→y X→x S→c I/Q→z`; drop length-1 leading axes.
  - Native chunk shape: leading axes 1, then `(tile_h, tile_w[, S])` for tiled or `(rows_per_strip, width[, S])` for strips, where the page-level dims are the trailing `len(page.shape)` axes.
  - Chunk get: leading coords → page index via `np.ravel_multi_index`; `(r, c)` → segment `r * page.chunked[1] + c`; `pread` `dataoffsets[seg]`/`databytecounts[seg]` from an `FdCache` fd; if `page.compression == 1 and page.predictor == 1 and page.fillorder == 1`: pass through raw bytes padded with zeros to the full chunk byte length, codecs `[bytes(endian=file byteorder)]`; else `page.decode(raw, seg)[0]` → squeeze to chunk dims → `pad_chunk` → `encode_chunk` with `OUTPUT_CODECS`; array `zarr.json` declares the chain actually emitted. Store marks such arrays in `Opened.transcoded: set[str]` of array paths (add field to `Opened`, default empty) for the `X-Zarrbridge-Source` header.
  - Stored pyramids: `series.levels[k]` → dataset `str(k)`, factor per axis = `round(shape0/shape_k)`, transforms via `level_transforms`.
  - Scale/units (`meta.py`): OME-XML `Pixels.PhysicalSize{X,Y,Z}` + `*Unit` (map `µm→micrometer`, `nm→nanometer`, `mm→millimeter`); ImageJ `spacing`/`unit` + XResolution; else scale 1.0, no unit.
  - Identity: `header_hash = "sha256:" + sha256(first 64 KiB + struct.pack("<%dq", offsets) of all page offsets)`; `extras={"pages": n, "series": n}`.
  - Classification: `servable_2d` if native chunk has leading extent 1 on an axis of length > 1 (plane-chunked stack); `unservable("compressed chunk > limits.max_chunk_bytes")` if any `databytecounts` max > limit and compressed; `servable` otherwise. `reader_reason` string supplied by the registry.

- [ ] **Step 1: Write failing tests**

```python
def test_lzw_strips_transcoded_to_zstd(tmp_path):
    a = np.arange(5*100*120, dtype="<u2").reshape(5,100,120)
    p = make_tiff(tmp_path/"a.tif", a, compression="lzw", rowsperstrip=32)
    op = TiffReader().open(p, cfg)
    meta = json.loads(get(op.store, "0/zarr.json"))
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1,32,120] and [c["name"] for c in meta["codecs"]] == ["bytes","zstd"]
    assert op.classification.state == "servable_2d"
    last = zstandard.decompress(get(op.store, "0/c/4/3/0")); assert len(last) == 32*120*2   # short strip padded
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)

def test_uncompressed_bigendian_passthrough(tmp_path):
    p = make_tiff(tmp_path/"be.tif", a, byteorder=">", rowsperstrip=32)
    meta = json.loads(get(op.store, "0/zarr.json")); assert meta["codecs"] == [{"name":"bytes","configuration":{"endian":"big"}}]
    assert get(op.store, "0/c/0/0/0") == a[0,:32].astype(">u2").tobytes()
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)

def test_tiled_zlib(tmp_path): ...  # tile=(32,32) → chunk [1,32,32]; edge tile padded; zarr reads back
def test_imagej_hyperstack_axes(tmp_path): ...  # TZCYX → axes names ["z","c","y","x"] after dropping T=1; shape (5,2,100,120)
def test_ome_pyramid_and_series(tmp_path):
    # 2 series, series 0 with one SubIFD level → images ["0","1"]; "0" levels ["0","1"]; dataset 1 scale == [2.0, 1.0, 1.0] for PhysicalSizeX=0.5 → wait: scale0 [2.0,0.5,0.5], level1 [2.0,1.0,1.0], translation [0,0.25,0.25]
    assert ds[1]["coordinateTransformations"] == [{"type":"scale","scale":[2.0,1.0,1.0]},{"type":"translation","translation":[0.0,0.25,0.25]}]
def test_rgb_contig(tmp_path): ...  # YXS → axes y x c, chunk (rows, width, 3)
```

- [ ] **Step 2: Run** → fails. **Step 3: Implement.** **Step 4: Run** → passes. **Step 5: Commit** `feat: tiff reader`.

---

### Task 5: refkv reader and reader registry

**Files:**
- Create: `zarrbridge/readers/refkv_reader.py`, `zarrbridge/readers/registry.py`
- Test: `tests/test_refkv_reader.py`, `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `class RefkvReader` (`name="refkv"`, `version=1`): `sniff(path, head)` = `refkv.ir.sniff(head[:552]) is not None`. `open(path, config, index_path: Path | None = None)`: if `index_path` exists and its sidecar `index.json` `{size, mtime_ns, header_hash, refkv_version}` matches the current file → `rs = refkv.open(str(index_path))`; else `rs = zarrify(str(path.resolve()), connections=config.readers.refkv.connections)` (catch `refkv.zarrify.Rejected`, `NotImplementedError`, `ValueError` → re-raise as `ReaderRejected(reason)` defined in `readers/base.py`). `store = refkv.zarr.store(rs)` (import `refkv.zarr` first). Images: root `zarr.json` has `ome.multiscales` → `[""]`; has `ome["bioformats2raw.layout"]` → read `OME/zarr.json` `ome.series`; each image's `ImageInfo` from its `multiscales[0]` and dataset-0 array meta. `identity.header_hash = sha256(first 64 KiB)`; `extras = {"pins": [{"url", "size", "etag", "modified_not_after"} for s in rs.sources if s.url], "format": rs.info["format"]}`; pins re-verified against `os.stat` (size) on open → mismatch raises `ReaderRejected("pin mismatch")`. `save_index(opened, index_path)` writes vzip + `index.json`. `Opened.transcoded` = array paths whose codec names ⊄ `ALLOWLIST`. Classification like tiff (plane-chunked → `servable_2d`).
  - `registry.py`: `select(path: Path, config: Config) -> tuple[Reader, str]` implementing spec §5.5 (reads first 64 KiB once; MRC suffix+magic → `MrcReader` unless `chunk_mode=="native"` → refkv; TIFF magic → `prefer` order with fallback: try refkv `open`, on `ReaderRejected` fall back to tiff with reason `f"refkv rejected: {e}"`; else refkv sniff → refkv; else raise `ReaderRejected("no reader")`). `open_dataset_reader(path, config, index_path=None) -> Opened` does the try/fallback and fills `reader_reason`.

- [ ] **Step 1: Write failing tests**

```python
def test_refkv_ome_tiff_root_image(tmp_path):
    p = make_tiff(tmp_path/"a.ome.tif", a, ome=True, metadata={"axes":"ZYX"}, tile=(32,32))
    op = RefkvReader().open(p, cfg)
    assert [i.path for i in op.images] == [""] and op.images[0].native_chunks == (1,32,32) and op.images[0].levels == ["0"]
    assert op.identity.extras["pins"][0]["size"] == p.stat().st_size
    assert np.array_equal(zarr.open_group(op.store, mode="r")["0"][:], a)

def test_refkv_rejects_plain_stack(tmp_path):
    with pytest.raises(ReaderRejected, match="without OME-XML"): RefkvReader().open(make_tiff(tmp_path/"s.tif", a), cfg)

def test_index_roundtrip(tmp_path):
    op = RefkvReader().open(p, cfg); save_index(op, tmp_path/"index.vzip")
    op2 = RefkvReader().open(p, cfg, index_path=tmp_path/"index.vzip"); assert op2.reader_reason.startswith("index") and reads back

def test_select_falls_back_to_tifffile(tmp_path):
    op = open_dataset_reader(make_tiff(tmp_path/"s.tif", a), cfg); assert op.reader_name == "tiff" and "refkv rejected" in op.reader_reason
def test_select_mrc(tmp_path): assert open_dataset_reader(make_mrc(...), cfg).reader_name == "mrc"
def test_select_no_reader(tmp_path): (tmp_path/"x.bin").write_bytes(b"\0"*100); with pytest.raises(ReaderRejected): ...
def test_refkv_lzw_marked_transcoded(tmp_path): ...  # ome lzw → "0" in op.transcoded
```

- [ ] **Step 2: Run** → fails. **Step 3: Implement.** **Step 4: Run** → passes. **Step 5: Commit** `feat: refkv reader and reader registry`.

---

### Task 6: TranscodeStore

**Files:**
- Create: `zarrbridge/store/transcode.py`
- Test: `tests/test_transcode.py`

**Interfaces:**
- Produces: `class TranscodeStore(zarr.abc.store.WrapperStore)`: `TranscodeStore(inner: Store, allowlist: set[str] = ALLOWLIST, output_codecs=OUTPUT_CODECS, max_decodes: int = os.cpu_count())`. On `get` of a key ending `zarr.json` whose node is an array with a codec name outside the allowlist: return the metadata with `codecs` replaced by `output_codecs` (cache parsed original meta per array path in `self._meta`). On a chunk `get` for such an array: `inner.get` full chunk → `decode_chunk` with the original meta → `encode_chunk` → apply `byte_range`. Bounded by `asyncio.Semaphore(max_decodes)`. `transcoded_arrays -> set[str]` (paths seen so far); `is_transcoded(key) -> bool` for the header layer. Metadata lookups for a chunk key whose array meta is not yet cached are loaded from `inner` (`<array>/zarr.json`) — one extra small read, then memoised.

- [ ] **Step 1: Write failing test**

```python
async def test_transcode_lzw_ome_tiff(tmp_path):
    op = RefkvReader().open(make_tiff(tmp_path/"a.ome.tif", a, ome=True, metadata={"axes":"ZYX"}, compression="lzw", rowsperstrip=32), cfg)
    ts = TranscodeStore(op.store)
    meta = json.loads((await ts.get("0/zarr.json", proto)).to_bytes()); assert [c["name"] for c in meta["codecs"]] == ["bytes","zstd"]
    assert np.array_equal(zarr.open_group(ts, mode="r")["0"][:], a) and ts.is_transcoded("0/c/0/0/0")
    raw = RefkvReader().open(make_tiff(tmp_path/"r.ome.tif", a, ome=True, metadata={"axes":"ZYX"}, tile=(32,32)), cfg)
    ts2 = TranscodeStore(raw.store); assert (await ts2.get("0/c/0/0/0", proto)).to_bytes() == (await raw.store.get("0/c/0/0/0", proto)).to_bytes()
```

- [ ] **Step 2–5:** run, implement, run, commit `feat: TranscodeStore`.

---

### Task 7: Cache layout, fingerprint, locks

**Files:**
- Create: `zarrbridge/cache/__init__.py`, `zarrbridge/cache/layout.py`, `zarrbridge/cache/fingerprint.py`, `zarrbridge/cache/locks.py`
- Test: `tests/test_fingerprint.py`, `tests/test_locks.py`

**Interfaces:**
- Produces:
  - `layout.py`: `cache_dir_for(path: Path, config: Config) -> tuple[Path, str]` returning `(dir, mode)`; `beside` if `os.access(path.parent, W_OK)` (or mode forced), else `central` under `expanduser(central_root)`; central dirs get a `source.json` `{"path": abs}` written by the builder. Constants `FINGERPRINT="fingerprint.json"`, `BUILD_JSON="build.json"`, `LOCK="lock"`, `INDEX="index.vzip"`, `INFO="info.json"`. `cache_bytes(dir) -> int`.
  - `fingerprint.py`: `SCHEMA_VERSION=1`, `DERIVATION_VERSION=1` (changelog comment block as in mrc-ng CLAUDE.md), `Validity = Literal["valid","stale","outdated","incompatible","missing","building"]`. `build_fingerprint(opened: Opened, plan: dict, built_by: str, duration_s: float) -> dict` with exactly the spec §7.3 keys. `write_fingerprint(dir, fp)` (tmp → fsync → replace → dir fsync). `read_fingerprint(dir) -> dict | None`. `validate(dir: Path, opened: Opened) -> tuple[Validity, str | None]` in this order: no file → `("missing", None)` unless `lock` is held by a live builder → `("building", None)`; schema → `incompatible`; derivation or `reader.version`/`reader.name` or `refkv_version` (refkv reader only) → `outdated`; `source.size/mtime_ns/header_hash/pins` → `stale` with the mismatching field name; else `valid`. `fingerprint_digest(fp) -> str` = sha256 of canonical JSON [:16] (for ETags).
  - `locks.py`: `class BuildLock`: `acquire(dir, stale_seconds=6*3600) -> BuildLock | None` (`O_CREAT|O_RDWR` + `flock(LOCK_EX|LOCK_NB)`; on `BlockingIOError` read `host:pid:started_at`; if same host, pid dead, and age > stale → log warning, unlink, retry once; else return None). Writes `f"{hostname}:{pid}:{iso}"`. `release()`. `is_locked(dir) -> bool` (try non-blocking lock, release immediately). `lock_info(dir) -> tuple[str,int,str] | None`.

- [ ] **Step 1: Write failing tests**

```python
def test_validate_states(tmp_path, mrc_opened):
    d = tmp_path/"c"; d.mkdir(); assert validate(d, mrc_opened) == ("missing", None)
    fp = build_fingerprint(mrc_opened, plan={"factor":[2,2,2],"levels":2,"level_chunks":[[64,64,64]],"codec":"zstd:3","method":"mean"}, built_by="t", duration_s=1.0)
    write_fingerprint(d, fp); assert validate(d, mrc_opened) == ("valid", None)
    fp2 = dict(fp, schema_version=99); write_fingerprint(d, fp2); assert validate(d, mrc_opened)[0] == "incompatible"
    fp3 = dict(fp, derivation_version=99); write_fingerprint(d, fp3); assert validate(d, mrc_opened)[0] == "outdated"
    # append a byte to the source → reopen → stale on size
    with open(mrc_opened.identity.path, "ab") as f: f.write(b"\0")
    reopened = MrcReader().open(Path(mrc_opened.identity.path), cfg); write_fingerprint(d, fp)
    assert validate(d, reopened) == ("stale", "size")

def test_lock_contention(tmp_path):
    a = BuildLock.acquire(tmp_path); assert a is not None and BuildLock.acquire(tmp_path) is None and is_locked(tmp_path)
    a.release(); assert not is_locked(tmp_path)

def test_stale_lock_broken(tmp_path):
    (tmp_path/"lock").write_text(f"{socket.gethostname()}:999999:2000-01-01T00:00:00Z"); assert BuildLock.acquire(tmp_path, stale_seconds=1) is not None

def test_cache_dir_modes(tmp_path):
    assert cache_dir_for(tmp_path/"a.mrc", cfg) == (tmp_path/".zarrbridge"/"a.mrc", "beside")
    ro = tmp_path/"ro"; ro.mkdir(); ro.chmod(0o555); d, m = cache_dir_for(ro/"a.mrc", cfg); assert m == "central"
```

- [ ] **Step 2–5:** run, implement, run, commit `feat: cache layout, fingerprint, locks`.

---

### Task 8: Builder (plan, downsample, streaming build, `build` CLI entry)

**Files:**
- Create: `zarrbridge/build/__init__.py`, `zarrbridge/build/plan.py`, `zarrbridge/build/downsample.py`, `zarrbridge/build/builder.py`
- Test: `tests/test_downsample.py` (port mrc-ng's reference mean test), `tests/test_plan.py`, `tests/test_builder.py`

**Interfaces:**
- Consumes: `Opened`, `omezarr.py` helpers, Task 7.
- Produces:
  - `downsample.py`: `block_mean(arr: np.ndarray, factors: tuple[int,...]) -> np.ndarray` verbatim from mrc-ng `downsample.py` (reduceat, one division, half-away-from-zero rounding, clip; any ndim).
  - `plan.py`: `@dataclass Level(index: int, shape: tuple, chunks: tuple, factor: tuple[int,...] (cumulative), scale: list[float], translation: list[float])`; `choose_factor(image: ImageInfo, config, path) -> tuple[int,...]` per spec §9.2: `factor_by_glob` override → per-axis factors (length = ndim, 1 for non-spatial axes; `"2,2,1"` applies to the trailing z,y,x axes in that order); `auto`: 2_2_1 if `native_chunks[z]==1` or `is_stack` or z spacing > 2× xy spacing; 2_2_2 if 3D chunks and spacings within 10 %; else 2_2_1; 2D images get (2,2). `plan_levels(image, factor, min_axis=64, max_levels=8) -> list[Level]`: level 0 = native; iterate while any downsampled axis ≥ min_axis after dividing, `shape_k = ceil(shape_{k-1}/step)`, step per axis = factor axis value or 1 once that axis < min_axis (as mrc-ng `plan_scales`); chunks: if native chunk is plane-slab (chunk y == shape y or chunk x == shape x) → `(…1s…, min(512, shape_y_k), min(512, shape_x_k))`; else native chunk clamped to level shape; transforms via `level_transforms`. `plan_dict(levels, factor, codec, method) -> dict` for the fingerprint.
  - `builder.py`: `build(path: Path, config: Config, *, factor: str | None = None, levels: int | None = None, min_axis: int | None = None, force: bool = False, jobs: int = 1, report: Path | None = None, progress: Callable | None = None) -> int` returning exit codes `0, 3 (SKIPPED_LOCKED), 4 (SKIPPED_VALID), 5 (UNSERVABLE), 1`. Algorithm: open reader via registry; unservable → 5; `validate` valid and not force → 4; `BuildLock.acquire` → None → 3; remove existing `<image>/<k>` dirs and `fingerprint.json`; for each image: `src = zarr.open_array(TranscodeStore(opened.store), path=f"{img}/0")`; build level 1 in row blocks: for each z-block of `chunks_z` output planes and y-block of `chunks_y` output rows, read input region `[z*fz : (z+cz)*fz, y*fy : (y+cy)*fy, :]` via `opened.read_region` if present else `src[...]`; `block_mean` → write into `zarr.create_array(LocalStore(cache/<img>/1.partial), shape, chunks, dtype, codecs=OUTPUT_CODECS, fill 0)`; split x when the block exceeds `max_block_bytes=256 MiB` on output-chunk boundaries; `jobs>1` → `ThreadPoolExecutor` over z-blocks. Levels ≥ 2: cascade from `zarr.open_array(cache/<img>/<k-1>)`. Rename `<k>.partial → <k>`; write `<img>/info.json` = reader's group meta with `datasets` extended for levels 1..N and `attributes["zarrbridge"] = {"levels": "multiscale", "built_at": ...}`; for refkv reader also `save_index`; `fsync` tree; `write_fingerprint` last; `build.json` updated every 16 blocks with `{state, level, done, total, started_at, eta_s}` then final report `{state:"done", bytes_read, bytes_written, duration_s, levels}`; `report` path gets the same JSON. Central mode writes `source.json`. `nice`/`ionice` applied via `os.nice(config.build.nice)` at start.

- [ ] **Step 1: Write failing tests**

```python
def test_block_mean_matches_reference(): ...  # port tests/test_downsample.py from mrc-ng-server, add a 4-D case

def test_plan_auto_2d_chunks_gives_221_tiles():
    img = ImageInfo(path="", shape=(1000,4096,4096), dtype="uint16", axes=[z,y,x], scale=[1,1,1], native_chunks=(1,128,4096), codecs=[], levels=["0"])
    assert choose_factor(img, cfg, Path("a.tif")) == (1,2,2)
    lv = plan_levels(img, (1,2,2), min_axis=64, max_levels=8); assert lv[1].shape == (1000,2048,2048) and lv[1].chunks == (1,512,512) and lv[1].translation == [0.0, 0.5, 0.5]
    assert len(lv) == 7  # 4096→64 is 6 halvings
def test_plan_cubes_isotropic_gives_222(): ...  # mrc (64,64,64) chunk → (2,2,2); chunks stay (64,64,64) clamped at tiny levels
def test_plan_glob_override(): ...  # factor_by_glob {"*.st": "2,2,1"}

def test_build_mrc_parity_with_reference(tmp_path):
    p = make_mrc(tmp_path/"a.mrc", (36,20,12), fill=canonical)   # x,y,z
    assert build(p, cfg_min_axis_4) == 0
    cache = tmp_path/".zarrbridge"/"a.mrc"; assert (cache/"fingerprint.json").exists() and not list(cache.glob("*.partial"))
    lvl1 = zarr.open_array(LocalStore(cache), path="1"); data = mrcfile.open(p, permissive=True).data
    assert np.array_equal(lvl1[:], block_mean(data, (2,2,2)))   # shape (6,10,18)
    info = json.loads((cache/"info.json").read_text()); assert [d["path"] for d in info["attributes"]["ome"]["multiscales"][0]["datasets"]] == ["0","1","2"]
    assert build(p, cfg) == 4   # SKIPPED_VALID

def test_build_tiff_stack_221(tmp_path): ...  # (5,100,120) lzw strips → level 1 shape (5,50,60), chunks (1,50,60), identical to block_mean(a,(1,2,2))
def test_build_locked_returns_3(tmp_path): lock = BuildLock.acquire(cache_dir); assert build(p, cfg) == 3
def test_kill_mid_build_leaves_missing(tmp_path):   # subprocess `zarrbridge build` with SIGKILL after .partial appears → no fingerprint; validate == missing
def test_builder_memory_bounded(tmp_path):   # 512 MiB synthetic uint8 tiff stack built under RLIMIT_AS 400 MiB in a subprocess (scaled-down version of spec's 4 GiB/1 GiB; marked slow)
```

- [ ] **Step 2–5:** run, implement, run, commit `feat: streaming pyramid builder`.

---

### Task 9: Dataset, PyramidOverlay, MetadataStore, auto-build queue, library API

**Files:**
- Create: `zarrbridge/store/overlay.py`, `zarrbridge/store/metadata.py`, `zarrbridge/dataset.py`, `zarrbridge/build/queue.py`, `zarrbridge/api.py`
- Test: `tests/test_dataset.py`, `tests/test_queue.py`

**Interfaces:**
- Produces:
  - `overlay.py`: `PyramidOverlay(inner: Store, cache_dir: Path, images: list[str], is_valid: Callable[[], bool])`: for keys whose `level_of(array_path)[1] >= 1` and image in `images` and `is_valid()` → `LocalStore(cache_dir).get(key)`; else `inner.get`. Missing cached chunk → `None` (404), never compute.
  - `metadata.store.py`: `MetadataStore(inner, cache_dir, images, is_valid, reason: Callable[[], str | None])`: `get("<img>/zarr.json")` (`"zarr.json"` for root image) → valid: bytes of `cache/<img>/info.json`; else inner meta with `attributes["zarrbridge"] = {"levels": "native", "reason": reason()}` injected.
  - `dataset.py`: `class Dataset`: `Dataset.open(path: Path, config: Config) -> Dataset`; attrs `path, opened, cache_dir, cache_mode, store` (composed stack: `MetadataStore(PyramidOverlay(TranscodeStore(reader store)))`), `validity: Validity`, `mismatch: str | None`; `recheck()` re-validates if `recheck_seconds` elapsed using `os.fstat(opened.fd)` when `fd` is set (size/mtime change → `stale` immediately); `status() -> dict` exactly §8.4 (`levels.state`: `unservable` if classification unservable; `multiscale` if valid; `building` if lock held (progress from `build.json`); `queued` if `AutoBuildQueue.pending(cache_dir)` or size > `auto_max_bytes` with reason text; `failed` if `build.json.state == "failed"`; else `native`; `advice: ["xy_only"]` when `servable_2d`); `etag_for(key) -> str` (`W/"<fingerprint_digest>:<key>"` for cached levels, `W/"<identity.digest()>:<key>"` otherwise); `source_of(key) -> "native"|"cache"|"transcoded"`; `ensure_pyramid(mode: Literal["never","auto","now"], queue: AutoBuildQueue | None) -> None`; `close()`.
  - `queue.py`: `AutoBuildQueue(config)`: `enqueue(path: Path, cache_dir: Path) -> bool` (dedupe by cache_dir; spawn `[sys.executable, "-m", "zarrbridge", "build", str(path), "--cache-mode", mode]` via `subprocess.Popen` with `nice`, bounded by `slots`; a worker thread drains a `queue.Queue`); `pending(cache_dir) -> bool`; `snapshot() -> dict` for `/_health`; `shutdown()`.
  - `api.py`: `open(path, config=None) -> Dataset`; `build = zarrbridge.build.builder.build`; `create_app` (Task 10, imported lazily); `Fingerprint` namespace exposing `validate`, `read_fingerprint`, `SCHEMA_VERSION`, `DERIVATION_VERSION`. `zarrbridge/__init__.py` re-exports `open, build, create_app, Config, Dataset`.

- [ ] **Step 1: Write failing tests**

```python
def test_native_only_then_multiscale(tmp_path):
    p = make_mrc(tmp_path/"a.mrc", (36,20,12), fill=canonical)
    ds = Dataset.open(p, cfg_min4); meta = json.loads(get(ds.store, "zarr.json"))
    assert meta["attributes"]["zarrbridge"]["levels"] == "native" and ds.status()["levels"]["state"] == "native"
    assert get(ds.store, "1/c/0/0/0") is None            # 404, nothing spawned
    assert build(p, cfg_min4) == 0; ds.recheck(force=True)
    meta = json.loads(get(ds.store, "zarr.json")); assert len(meta["attributes"]["ome"]["multiscales"][0]["datasets"]) == 3
    assert ds.status()["levels"]["state"] == "multiscale" and ds.source_of("1/c/0/0/0") == "cache" and ds.etag_for("1/c/0/0/0").startswith('W/"')
    g = zarr.open_group(ds.store, mode="r"); assert g["1"].shape == (6,10,18)

def test_replaced_file_goes_stale(tmp_path):
    ... build, open, then rewrite the file with different data (same size) ... ds.recheck(force=True)
    assert ds.validity == "stale" and ds.status()["cache"]["mismatch"] in ("mtime_ns", "header_hash")
    assert "1" not in [d["path"] for d in json.loads(get(ds.store, "zarr.json"))["attributes"]["ome"]["multiscales"][0]["datasets"]]

def test_queue_dedupes_and_spawns(tmp_path, monkeypatch):  # monkeypatch Popen to record argv; enqueue twice → one spawn; pending() True until drained
def test_ensure_pyramid_auto_respects_max_bytes(tmp_path): ...  # auto_max_bytes=0 → not enqueued, status reason mentions `zarrbridge build`
```

- [ ] **Step 2–5:** run, implement, run, commit `feat: Dataset, overlay/metadata stores, auto-build queue, library API`.

---

### Task 10: HTTP server

**Files:**
- Create: `zarrbridge/server/__init__.py`, `zarrbridge/server/app.py`, `zarrbridge/server/resolver.py`, `zarrbridge/server/headers.py`, `zarrbridge/server/browse.py`
- Test: `tests/test_server.py`, `tests/test_trusted_path.py`

**Interfaces:**
- Produces:
  - `resolver.py`: `DatasetResolver(config, queue)`: `resolve(root: str, path: str) -> tuple[Dataset, str]` (longest prefix of `path` segments under `config.roots[root]` that is a regular file; containment via `Path.resolve()` must stay under the root unless `follow_symlinks`; raises `KeyError`/`FileNotFoundError`); `resolve_abs(abs_path: str) -> Dataset` for trusted mode (checked against `allow_prefixes` with `fnmatch` on the absolute path); LRU of `limits.max_open_datasets` with idle close; `open_count`.
  - `headers.py`: `cache_control(ds, key, immutable: bool) -> str` (metadata `no-cache`; native chunks `public, max-age=3600, must-revalidate`; cached `public, max-age=86400, must-revalidate`; immutable → `public, max-age=31536000, immutable`); `read_strategy(ds, key) -> str | None` (`rowwise|spanwise` from the mrc reader's `choose_strategy` on the chunk's x extent; `reference` for refkv; `None` for tiff).
  - `app.py`: `create_app(config: Config) -> Starlette`. Routes in order: `GET /_health`, `GET /_status/{root}/{path:path}` (or `GET /_status` in trusted mode), `GET /` and `GET /{root}/` + `/{root}/{path:path}/` browse when `server.browse` (trailing slash or no zarr key, directory listing), `GET|HEAD /{root}/{path:path}`. Key handler: resolve → `ds.recheck()` → `store.get(key, byte_range)` (Range header → `RangeByteRequest`/`OffsetByteRequest`/`SuffixByteRequest`; 206 + `Content-Range`) → 404 on `None` → headers `ETag`, `Cache-Control`, `X-Zarrbridge-Source`, `X-Zarrbridge-Read`, `Content-Type` (`application/json` for `zarr.json` else `application/octet-stream`); `If-None-Match` equal → 304; HEAD returns headers with `Content-Length`. Reject keys with `""`, `.`, `..` segments or NUL → 404 (same rule as zarr-http-server). Middleware: token (`Authorization: Bearer` or `?token=` → else 401), CORS (`CORSMiddleware` with `expose_headers=["Content-Range","ETag","X-Zarrbridge-Source","X-Zarrbridge-Read"]`, `allow_headers=["Range","If-None-Match"]`), JSON access log (`zarrbridge.access`: `path, key, status, bytes, duration_ms, source, read_strategy`). Trusted mode: `config.server.trusted_path_header` set → every data/status request requires the header (400 without), URL path = zarr key only, no `{root}`. Startup refuses non-loopback host without token unless `server.insecure`. `ensure_pyramid("auto")` called on first open when `build.auto`. Lifespan closes resolver and queue.
  - `browse.py`: `render_listing(root, relpath, entries, base_url) -> str` HTML: directories as links, files with a `neuroglancer` link (`https://neuroglancer-demo.appspot.com/#!` + JSON `{"layers":[{"type":"image","source":"zarr3://<url>"}]}` URL-encoded) and `vizarr` link (`https://hms-dbmi.github.io/vizarr/?source=<url>`), filtered to files whose `registry.select` sniff succeeds (reads 64 KiB per file; bounded to 500 entries).

- [ ] **Step 1: Write failing tests**

```python
def test_serve_mrc_end_to_end(tmp_path):
    p = make_mrc(tmp_path/"data"/"a.mrc", (80,60,40)); app = create_app(cfg_with_root("data", tmp_path/"data", build_auto=False))
    c = TestClient(app)
    r = c.get("/data/a.mrc/zarr.json"); assert r.status_code == 200 and r.headers["cache-control"] == "no-cache" and r.headers["etag"].startswith('W/"')
    r2 = c.get("/data/a.mrc/zarr.json", headers={"If-None-Match": r.headers["etag"]}); assert r2.status_code == 304
    r = c.get("/data/a.mrc/0/c/0/0/0"); assert r.status_code == 200 and len(r.content) == 64**3*2 and r.headers["x-zarrbridge-source"] == "native" and r.headers["x-zarrbridge-read"] == "spanwise"
    assert c.get("/data/a.mrc/0/c/9/9/9").status_code == 404 and c.get("/data/a.mrc/1/c/0/0/0").status_code == 404
    assert c.head("/data/a.mrc/0/c/0/0/0").headers["content-length"] == str(64**3*2)
    r = c.get("/data/a.mrc/0/c/0/0/0", headers={"Range": "bytes=0-9"}); assert r.status_code == 206 and len(r.content) == 10
    assert c.get("/data/../etc/passwd").status_code == 404 and c.get("/data/a.mrc/../a.mrc/zarr.json").status_code == 404
    s = c.get("/_status/data/a.mrc").json(); assert s["reader"] == "mrc" and s["levels"]["state"] == "native"
    assert c.get("/_health").json()["ok"] is True

def test_zarr_python_over_http(tmp_path, live_server):   # uvicorn in a thread; zarr.open_group(FsspecStore/http url) reads level 0 equal to mrcfile data

def test_token_required(): 401 without, 200 with `?token=`
def test_symlink_escape_404(tmp_path): ...
def test_browse_lists_files(): "a.mrc" and "neuroglancer-demo" in html

def test_trusted_path_mode(tmp_path):
    app = create_app(cfg(trusted_path_header="X-Fg-Resolved-Path", allow_prefixes=[str(tmp_path)+"/**"], browse=False))
    c = TestClient(app); assert c.get("/zarr.json").status_code == 400
    assert c.get("/zarr.json", headers={"X-Fg-Resolved-Path": str(p)}).status_code == 200
    assert c.get("/zarr.json", headers={"X-Fg-Resolved-Path": "/etc/passwd"}).status_code == 403
    assert c.get("/_status", headers={"X-Fg-Resolved-Path": str(p)}).json()["reader"] == "mrc"
```

- [ ] **Step 2–5:** run, implement, run, commit `feat: HTTP server`.

---

### Task 11: CLI (`serve build status prune index inspect`) and prune

**Files:**
- Create: `zarrbridge/cli.py`, `zarrbridge/__main__.py`, `zarrbridge/cache/prune.py`, `zarrbridge/cache/status.py`
- Test: `tests/test_cli.py`, `tests/test_prune.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int` (argparse). Global flags `--config PATH`, `--cache-mode`, `--central-root`, `--log-format json|text`, `-v`. Subcommands and flags per spec §9.1/§11: `serve [paths...] --host --port --uds --token --insecure --cors-origin (repeat) --browse/--no-browse --trusted-path-header --allow-prefix (repeat) --build-auto on|off --immutable-chunks --follow-symlinks` (positional dirs become roots named by basename; `--root name=path` repeatable); `build <file|dir> --factor --levels --min-axis --codec --jobs --force --report` (dir → every sniffable file, aggregate exit code: 0 if all built/skipped-valid, else the first non-zero); `status <file|dir>` (table: path, reader, classification, levels state, cache bytes, fingerprint state, mismatch); `prune <dir|central_root> --older-than 90d --max-bytes N --stale --dry-run`; `index <file>` (refkv only, exit 5 otherwise); `inspect <file> [--mbps 200]` (reader, reason, grid, codecs, transcode decision, planned levels, estimated cache bytes = Σ level bytes / 3 (zstd guess), ETA = source bytes / mbps). `status.py`: `dataset_status_row(path, config) -> dict`; `prune.py`: `prune(root: Path, config, *, older_than_s: float | None, max_bytes: int | None, stale_only: bool, dry_run: bool) -> list[Path]` (skip locked; delete `stale|outdated|incompatible` and orphaned central caches whose `source.json` path is gone; then LRU by `fingerprint.json` atime until under `max_bytes`).

- [ ] **Step 1: Write failing tests**

```python
def test_cli_build_status_prune(tmp_path, capsys):
    p = make_mrc(tmp_path/"a.mrc", (36,20,12)); assert main(["build", str(p), "--min-axis", "4"]) == 0
    assert main(["status", str(tmp_path)]) == 0 and "multiscale" in capsys.readouterr().out
    with open(p, "ab") as f: f.write(b"\0")
    assert main(["status", str(tmp_path)]) == 0 and "stale" in capsys.readouterr().out
    assert main(["prune", str(tmp_path), "--stale"]) == 0 and not (tmp_path/".zarrbridge"/"a.mrc").exists()
def test_cli_inspect_tiff(tmp_path, capsys): ... "tiff" in out and "servable_2d" in out
def test_cli_build_dir_exit_codes(tmp_path): ...
```

- [ ] **Step 2–5:** run, implement, run, commit `feat: CLI and prune`.

---

### Task 12: Conformance, parity, request-path invariant, docs and deploy

**Files:**
- Create: `tests/test_conformance.py`, `tests/test_parity_mrcng.py`, `tests/test_request_path.py`, `deploy/Dockerfile`, `deploy/zarrbridge.service`, `zarrbridge.example.toml`; update `README.md`
- Modify: `zarrbridge/readers/mrc/read.py` (add a module-level `_READ_COUNTER` hook: `pread_exact` increments `(calls, bytes)` when `ZB_INSTRUMENT_READS` is set — test-only instrumentation, zero cost otherwise)

**Interfaces:**
- Consumes everything.

- [ ] **Step 1: Write the tests**

```python
@pytest.mark.parametrize("fixture", ["mrc_cubes", "tiff_lzw_stack", "tiff_tiled", "ome_pyramid", "refkv_ome", "imagej"])
def test_ngff_05_conformance(fixture, tmp_path):
    ds = open_fixture(fixture, tmp_path); g = zarr.open_group(ds.store, mode="r")
    for img in ds.opened.images:
        Image.from_zarr(g if img.path == "" else g[img.path])     # ome_zarr_models.v05.image.Image
        for lvl in img.levels: _ = g[img.path][lvl] if img.path else g[lvl]; assert _.shape == tuple(json.loads(...)["shape"])

def test_parity_level0_and_pyramid_with_mrc_ng_server(tmp_path):
    # Requires mrcng importable (sibling checkout on sys.path; skip otherwise). Build with both: mrc-ng omezarr layout, zarrbridge cubes.
    # level 0: byte-identical chunk bodies for every chunk key.  level ≥1: zarrbridge chunks are zstd; decompress and compare to mrc-ng raw bytes → identical.
    # metadata: identical except documented diffs: axes unit (angstrom vs nanometer, scale ×10), attribute namespace (mrc/zarrbridge vs mrcng), omero block present.

def test_request_path_invariant_mrc(tmp_path, monkeypatch):
    monkeypatch.setenv("ZB_INSTRUMENT_READS", "1"); ... get one 64³ chunk; calls, nbytes = read_counter()
    assert calls <= 64 and nbytes <= 2 * 64**3 * 2 + (63 * nx + 64) * 2
def test_request_path_invariant_tiff(tmp_path): # one chunk GET → exactly one pread of databytecounts[seg]
def test_missing_cached_level_spawns_nothing(tmp_path, monkeypatch): # Popen monkeypatched to fail loudly; GET /…/1/c/0/0/0 → 404
```

- [ ] **Step 2: Run, fix anything they expose, run.**
- [ ] **Step 3: README** (install, `zarrbridge serve`, `build`, `status`, cache layout, Fileglancer contract summary, parity diffs list), `deploy/Dockerfile` (python:3.12-slim, `pip install .`, `CMD zarrbridge serve /data --host 0.0.0.0 --insecure`), `deploy/zarrbridge.service` (`systemd --user` unit), `zarrbridge.example.toml` = spec §10 block.
- [ ] **Step 4: Commit** `test: conformance, parity, request-path invariants; docs and deploy`.

---

## Deliberately deferred (spec v1.1 or non-CI)

`--shard`, tifffile index persistence, `build.method=mode`, bffile, resume builds, `zarrita` node conformance in CI, `benchmark.py` port (bench harness is one httpx loop; added only if asked), `refkv.compose` (not in 0.2.0), `limits.chunk_cache_bytes` LRU (kernel page cache covers v1; status reports 0).
