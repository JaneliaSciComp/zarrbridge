# zarrbridge

Virtual **OME-Zarr 0.5** serving for legacy image files. Point it at a directory of MRC, TIFF, OME-TIFF,
CZI, ND2, LIF, NDPI, SVS, IMS, DICOM, NIfTI, NRRD or FITS files and any OME-Zarr viewer (Neuroglancer, vizarr,
napari-ome-zarr, zarr-python) opens them over plain HTTP. Nothing is converted or copied: a chunk request is a
seek and a read of the file's own storage unit. Multiscale levels are built once, in a separate process, into a
sidecar cache guarded by a fingerprint. Design: `docs/specs/zarrbridge-spec.md`.

## Install

[pixi](https://pixi.sh) manages the environment, as in fileglancer and x2s3:

```bash
pixi install                 # default env: python 3.12 + zarrbridge (editable); refkv pinned to 0.2.0
pixi run zarrbridge --help
pixi run -e test test        # pytest
pixi run -e test test-parity # parity tests against a sibling ~/dev/mrc-ng-server checkout
pixi run dev-launch /path/to/data   # serve with browse page and CORS on
```

Plain pip works too: `pip install -e '.[test]'` on Python >= 3.12.

## Use

```bash
# inside `pixi shell`, or prefix each command with `pixi run`
zarrbridge serve /path/to/data --port 8800 --cors-origin '*'   # browse page at http://127.0.0.1:8800/
zarrbridge inspect file.tif          # reader, grid, codecs, transcode decision, planned pyramid
zarrbridge build  file.tif|dir       # exit 0 built, 3 locked, 4 already valid, 5 unservable
zarrbridge status dir                # reader, classification, levels state, fingerprint state
zarrbridge prune  dir --stale        # drop stale/outdated/orphaned caches; --older-than 90d --max-bytes 1TiB
zarrbridge index  file.czi           # write the refkv index only
```

URLs are `http://host/<root>/<path-to-file>/<zarr key>`; `/_status/<root>/<path>` is the status JSON and
`/_health` the liveness check. Configuration: `zarrbridge.toml` (see `zarrbridge.example.toml`), `ZB_*`
environment variables, CLI flags; precedence CLI > env > file.

```python
import zarrbridge, zarr
ds = zarrbridge.open("/data/tomo.mrc")          # reader selected, cache validated
g = zarr.open_group(ds.store, mode="r")         # the composed zarr v3 store
ds.status()                                     # the /_status dict
ds.ensure_pyramid("now")                        # run the builder in-process (CLI/jobs only)
app = zarrbridge.create_app(zarrbridge.Config.load("zarrbridge.toml"))
```

## How it works

- **Readers** (`zarrbridge/readers/`): `mrc` (64³ cubes over the uncompressed file, port of mrc-ng-server),
  `tiff` (tifffile; one strip or tile per chunk, decoded and re-encoded as zstd when the codec is not
  client-decodable), `refkv` (everything refkv zarrifies, references resolved by refkv). Selection:
  `.mrc/.rec/.st/.ali/.map` → mrc; TIFF → refkv, falling back to tifffile when refkv rejects (ImageJ stacks
  without OME-XML, short-tailed Deflate/zstd strips, ...); anything refkv sniffs → refkv.
- **Store stack** (`zarrbridge/store/`): `TranscodeStore` (codec allowlist) → `PyramidOverlay` (levels ≥ 1 from
  the cache while the fingerprint is valid) → `MetadataStore` (the builder's `info.json` is the authority).
- **Cache** (`zarrbridge/cache/`): `<dir>/.zarrbridge/<file>/` beside the data when writable, else
  `~/.cache/zarrbridge/pyramids/<sha>/`. `fingerprint.json` is written last after fsync; any mismatch
  (source size/mtime/header hash, reader version, derivation version, schema) drops the dataset to native-only
  and the status says which field. The server never writes the cache.
- **Builder** (`zarrbridge/build/`): streams level 0 through the same store the server uses, block-mean
  downsampling in row blocks with bounded memory, cascading level k+1 from level k on disk, `.partial`
  directories renamed into place, `flock` lock, `build.json` progress. Auto-build spawns
  `zarrbridge build` subprocesses (never in-process), deduped by cache dir, bounded by `build.slots`.
- **Server** (`zarrbridge/server/`): Starlette; `ETag` + `If-None-Match` → 304, `Cache-Control` per level,
  byte ranges, `X-Zarrbridge-Source: native|cache|transcoded`, `X-Zarrbridge-Read: rowwise|spanwise|reference|segment`,
  optional bearer token, CORS, JSON access log. Trusted-path mode for Fileglancer:
  `--uds sock --trusted-path-header X-Fg-Resolved-Path --allow-prefix '/groups/**'`.

## Parity with mrc-ng-server

Level-0 cube chunks are byte-identical; built chunks are identical over the clamped extent. Documented
metadata diffs: axes carry `unit: angstrom` with scales in Å (mrc-ng-server used nanometer), the header mirror is
`attributes["mrc"]`, an `omero.window` from dmin/dmax is added, and chunks are zstd. `tests/test_parity_mrcng.py`
runs when a sibling `mrc-ng-server` checkout is importable.

## Bumping the derivation version

If a change can alter what a build produces (downsampling, padding, translation convention, level layout,
codec), bump `DERIVATION_VERSION` in `zarrbridge/cache/fingerprint.py`, or every existing cache keeps being
served as valid.

## Not in v1

`--shard`, tifffile index persistence, label-aware (`mode`) downsampling, bffile reader, resume builds,
`refkv.compose` overlays (not in refkv 0.2.0), an in-process encoded-chunk LRU (`limits.chunk_cache_bytes`
is accepted but the kernel page cache does the work).
