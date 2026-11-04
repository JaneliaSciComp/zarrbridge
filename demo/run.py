"""zarrbridge demo: fetch public sample files in every supported format (once), then serve them with an index
page of Neuroglancer / vizarr links.

    pixi run demo                      # data in $ZB_DEMO_DIR, default /scratch/$USER/zarrbridge-demo or ~/.cache/zarrbridge-demo
    pixi run demo --fetch-only         # just download
    pixi run demo --port 8822          # fixed port (default: a free random port), always bound to 0.0.0.0
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

OME = "https://downloads.openmicroscopy.org/images"
MRCFILE = "https://raw.githubusercontent.com/ccpem/mrcfile/64b0fb9ce0d921b902a06fe94d43e8d642b33d7d/tests/test_data"

# (relative target path, source URL, what it exercises). `.gz` / `.zip` sources are unpacked after download.
MANIFEST: list[tuple[str, str, str]] = [
    # --- Janelia S3 (s3.janelia.org, anonymous) ---
    ("janelia/test.tiff", "https://s3.janelia.org/scicompsoft/b2rtests/test.tiff", "Janelia S3: plain TIFF stack (bioformats2raw test input)"),
    ("janelia/test_rotated.ome.tiff", "https://s3.janelia.org/scicompsoft/b2rtests/test_rotated.ome.tiff", "Janelia S3: OME-TIFF"),
    # --- MRC (mrc reader, 64^3 cubes over uncompressed data) ---
    ("mrc/emd_8117.map", "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8117/map/emd_8117.map.gz", "EMDB cryo-EM map (3D volume, float32)"),
    ("mrc/emd_3197.map", f"{MRCFILE}/EMD-3197.map", "EMDB map, tiny (mrcfile test data)"),
    ("mrc/epu2.9_example.mrc", f"{MRCFILE}/epu2.9_example.mrc", "EPU micrograph (single-plane MRC)"),
    # --- TIFF via tifffile (refkv rejects these: ImageJ / libtiff variants) ---
    ("tiff/mitosis.tif", "https://imagej.net/images/mitosis.tif", "ImageJ 5D hyperstack (TZCYX), uncompressed"),
    ("tiff/flybrain/flybrain.tif", "https://imagej.net/images/flybrain.zip", "ImageJ RGB stack"),
    ("tiff/libtiff/cramps-tile.tif", f"{OME}/TIFF/libtiff/cramps-tile.tif", "libtiff tiled sample"),
    ("tiff/libtiff/caspian.tif", f"{OME}/TIFF/libtiff/caspian.tif", "libtiff sample"),
    ("tiff/condensation/A1.pattern2.tif", f"{OME}/TIFF/condensation/A1.pattern2.tif", "plain TIFF stack"),
    # --- OME-TIFF (refkv) ---
    ("ome-tiff/tubhiswt-3D/tubhiswt_C0.ome.tif", f"{OME}/OME-TIFF/2016-06/tubhiswt-3D/tubhiswt_C0.ome.tif", "multi-file OME-TIFF, channel 0 (tifffile companion fallback)"),
    ("ome-tiff/tubhiswt-3D/tubhiswt_C1.ome.tif", f"{OME}/OME-TIFF/2016-06/tubhiswt-3D/tubhiswt_C1.ome.tif", "multi-file OME-TIFF, channel 1"),
    ("ome-tiff/retina_large.ome.tiff", f"{OME}/OME-TIFF/2016-06/sub-resolutions/Z-stack/retina_large.ome.tiff", "OME-TIFF with stored sub-resolutions (Z-stack)"),
    # --- Whole-slide: SVS / NDPI / QPTIFF (refkv; JPEG transcoded to zstd) ---
    ("svs/CMU-1-Small-Region.svs", "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs", "Aperio SVS, JPEG tiles (OpenSlide test data)"),
    ("ndpi/test3-DAPI.ndpi", f"{OME}/Hamamatsu-NDPI/manuel/test3-DAPI%202%20%28387%29%20.ndpi", "Hamamatsu NDPI, fluorescence"),
    ("ndpi/ki67_lymphoma.ndpi", "https://zenodo.org/api/records/18302140/files/ki67_lymphoma.ndpi/content", "Hamamatsu NDPI, brightfield pyramid (Zenodo 18302140)"),
    ("qptiff/2025-07-15_13_13_34.qptiff", "https://zenodo.org/api/records/16569043/files/2025-07-15_13_13_34.qptiff/content", "Akoya/PerkinElmer QPTIFF (Zenodo 16569043)"),
    # --- Microscope vendor formats (refkv) ---
    ("czi/Plate1-Blue-A-02-Scene-1-P2-E1-01.czi", "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0011-ledesmafernandez-dad4/20150826-peter_thorpe/Stinger%20plates-extra%20images%20screen%20T34%20x%20TS-more%20repeats/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-02-Scene-1-P2-E1-01.czi", "Zeiss CZI (IDR idr0011)"),
    ("nd2/HE_002.nd2", f"{OME}/ND2/zenodo-12734440/HE_002.nd2", "Nikon ND2 (Zenodo 12734440)"),
    ("nd2/MRAP1_KO_DN_10X03.nd2", f"{OME}/ND2/zenodo-10277961/MRAP1%20KO%20DN_10X03.nd2", "Nikon ND2 (Zenodo 10277961)"),
    ("lif/PR2729_frameOrderCombinedScanTypes.lif", f"{OME}/Leica-LIF/michael/PR2729_frameOrderCombinedScanTypes.lif", "Leica LIF, several images"),
    ("lif/Project007.lif", "https://zenodo.org/api/records/6606445/files/Project007.lif/content", "Leica LIF (Zenodo 6606445)"),
    ("ims/Convallaria_3C_1T_confocal.ims", f"{OME}/Imaris-IMS/davemason/Convallaria_3C_1T_confocal.ims", "Imaris IMS (HDF5), 3 channels"),
    # --- Medical / astro (refkv) ---
    ("dicom/MR-MONO2-16-head.dcm", f"{OME}/DICOM/samples/MR-MONO2-16-head.dcm", "DICOM, single frame MR"),
    ("dicom/idc-8afac71d.dcm", "https://idc-open-data.s3.amazonaws.com/d8d93a71-dfc8-43da-90ca-11e9aaf45cfe/8afac71d-a7c8-4ca4-b4c0-2e0261dfa384.dcm", "DICOM (Imaging Data Commons)"),
    ("nifti/DT_HIGH_32DIR_SENSE_1201.nii", "https://raw.githubusercontent.com/neurolabusc/dcm_qa_philips/74efdbc01eb62540fbb702787c3a7a2c0e22f9eb/Ref/DT_HIGH_32DIR_SENSE_1201.nii", "NIfTI-1, 4D diffusion"),
    ("nrrd/BallBinary30x30x30_gz.nrrd", "https://raw.githubusercontent.com/mhe/pynrrd/96bf87b78bde03987c09346ceffe0bb85a8f4214/nrrd/tests/data/BallBinary30x30x30_gz.nrrd", "NRRD, gzip encoded"),
    ("nrrd/BallBinary30x30x30.nhdr", "https://raw.githubusercontent.com/mhe/pynrrd/96bf87b78bde03987c09346ceffe0bb85a8f4214/nrrd/tests/data/BallBinary30x30x30.nhdr", "NRRD detached header (.nhdr + .raw)"),
    ("nrrd/BallBinary30x30x30.raw", "https://raw.githubusercontent.com/mhe/pynrrd/96bf87b78bde03987c09346ceffe0bb85a8f4214/nrrd/tests/data/BallBinary30x30x30.raw", "NRRD detached raw data"),
    ("fits/HorseHead.fits", "https://imagej.net/images/HorseHead.fits", "FITS image (Horsehead nebula)"),
    ("fits/test0.fits", "https://raw.githubusercontent.com/astropy/astropy/7d7cf6d2d3e8b1e381e980278d071067b4b25241/astropy/io/fits/tests/data/test0.fits", "FITS, multiple HDUs (astropy test data)"),
]

NEUROGLANCER = "https://neuroglancer-demo.appspot.com/#!"
VIZARR = "https://hms-dbmi.github.io/vizarr/?source="


def default_dir() -> Path:
    if os.environ.get("ZB_DEMO_DIR"):
        return Path(os.environ["ZB_DEMO_DIR"])
    scratch = Path("/scratch") / os.environ.get("USER", "user")
    return (scratch if scratch.is_dir() else Path.home() / ".cache") / "zarrbridge-demo"


def fetch(data: Path) -> None:
    for rel, url, _ in MANIFEST:
        target = data / rel
        if target.exists() and (target.with_suffix(target.suffix + ".ok")).exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch {url}\n   -> {target}", file=sys.stderr, flush=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zarrbridge-demo"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp_path, "wb") as out:
                shutil.copyfileobj(r, out, 16 << 20)
            if url.endswith(".gz"):
                with gzip.open(tmp_path, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
                tmp_path.unlink()
            elif url.endswith(".zip"):
                with zipfile.ZipFile(tmp_path) as z:
                    member = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                tmp_path.unlink()
            else:
                os.replace(tmp_path, target)
            target.with_suffix(target.suffix + ".ok").write_text(url)
        except Exception as e:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            print(f"   FAILED: {e}", file=sys.stderr, flush=True)


def free_port() -> int:
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def ng_link(url: str, name: str, viewer_base: str | None) -> str:
    from zarrbridge.server.browse import neuroglancer_state
    state = {**neuroglancer_state(url, name), "layout": "4panel"}
    base = f"{viewer_base}#!" if viewer_base else NEUROGLANCER
    return base + quote(json.dumps(state, separators=(",", ":")), safe="")


def write_index(data: Path, base_url: str, root: str, config) -> Path:
    from zarrbridge.dataset import Dataset
    from zarrbridge.server.app import viewer_static_dir
    viewer_base = f"{base_url}/neuroglancer/" if viewer_static_dir(config) else None
    from zarrbridge.readers.base import ReaderRejected
    from zarrbridge.readers.registry import sniffable
    rows = []
    for rel, url, note in MANIFEST:
        p = data / rel
        if not p.exists() or not sniffable(p):
            continue
        try:
            ds = Dataset.open(p, config)
            st = ds.status()
            ds.close()
            imgs = "; ".join(f"{i['path'] or '/'} {i['shape']} {i['dtype']} {''.join(i['axes'])} chunks {i['native_chunks']}" for i in st["images"])
            state = f"{st['reader']} · {st['classification']}" + (f" · {', '.join(st['advice'])}" if st["advice"] else "")
        except ReaderRejected as e:
            imgs, state = "", f"unservable: {e}"
        href = f"{base_url}/{root}/{quote(rel)}"
        if state.startswith("unservable"):
            links = ""
        else:
            # one viewer link per image: viewers open an OME-Zarr image group, not a bioformats2raw collection
            parts = []
            for i in st["images"]:
                ihref = f"{href}/{quote(i['path'])}" if i["path"] else href
                label = f"image {i['path']}: " if i["path"] else ""
                parts.append(f'{label}<a href="{html.escape(ng_link(ihref, p.name, viewer_base))}" target="_blank">Neuroglancer</a> · '
                             f'<a href="{html.escape(VIZARR + quote(ihref, safe=""))}" target="_blank">vizarr</a> · '
                             f'<a href="{html.escape(ihref)}/zarr.json">zarr.json</a>')
            links = "<br>".join(parts) + f'<br><a href="{base_url}/_status/{root}/{quote(rel)}">status</a>'
        rows.append(f"<tr><td><code>{html.escape(rel)}</code><br><small>{html.escape(note)} · <a href='{html.escape(url)}'>source</a></small></td>"
                    f"<td>{html.escape(state)}<br><small>{html.escape(imgs)}</small></td><td>{links}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>zarrbridge demo</title>
<style>body{{font-family:system-ui,sans-serif;margin:2em;max-width:1400px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:.5em;vertical-align:top;text-align:left}}small{{color:#555}}code{{font-size:.9em}}</style></head>
<body><h1>zarrbridge demo</h1>
<p>Public sample files in every supported format, served live as OME-Zarr 0.5 at <code>{html.escape(base_url)}/{root}/&lt;file&gt;/</code>.
Nothing was converted: each chunk is read from the original file on request. Multiscale levels are built in the background
on first open; reload a <em>status</em> link to watch <code>levels.state</code> go <code>native → building → multiscale</code>.
Neuroglancer links open a client served from this origin, so no cross-site permission is needed. vizarr links open the
hosted hms-dbmi.github.io page: on a private network Chrome asks permission for that site to reach this server; allow it once.
<a href="{base_url}/{root}/">Directory listing</a> · <a href="{base_url}/_health">health</a></p>
<table><tr><th>file</th><th>reader · classification · images</th><th>open</th></tr>{''.join(rows)}</table></body></html>"""
    out = data / "index.html"
    out.write_text(page)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=default_dir())
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host-name", default=socket.getfqdn(), help="hostname to put in the printed URL and viewer links")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--access-log", action="store_true", help="JSON access log on stderr (one line per request)")
    ap.add_argument("--viewer-dir", default=os.environ.get("ZB_VIEWER_DIR", ""),
                    help="standalone Neuroglancer client build (neuroglancer/dist/client) served at /neuroglancer/")
    ap.add_argument("--ssl-certfile", default=os.environ.get("ZB_SSL_CERTFILE", "/opt/certs/cert.crt" if Path("/opt/certs/cert.crt").exists() else ""))
    ap.add_argument("--ssl-keyfile", default=os.environ.get("ZB_SSL_KEYFILE", "/opt/certs/cert.key" if Path("/opt/certs/cert.key").exists() else ""))
    args = ap.parse_args(argv)
    data = args.dir
    data.mkdir(parents=True, exist_ok=True)
    if not args.no_fetch:
        fetch(data)
    if args.fetch_only:
        return 0
    from zarrbridge.server.config import Config
    port = args.port or free_port()
    ssl = bool(args.ssl_certfile)
    base_url = f"{'https' if ssl else 'http'}://{args.host_name}:{port}"
    root = data.name
    config = Config.load(env={}, overrides={"roots": {root: str(data)},
                                            "server": {"host": "0.0.0.0", "port": port, "insecure": True, "cors_origins": ["*"]},
                                            "build": {"auto": True, "slots": 2}})
    config.server.viewer_dir = args.viewer_dir
    index = write_index(data, base_url, root, config)
    config.server.index_html = str(index)
    print(f"\nzarrbridge demo: {base_url}/\n", flush=True)
    if args.access_log:
        import logging
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", stream=sys.stderr)
        logging.getLogger("zarrbridge.access").setLevel(logging.INFO)
        logging.getLogger("zarrbridge.build").setLevel(logging.INFO)
    import uvicorn
    from zarrbridge.server.app import create_app
    kw = {"ssl_certfile": args.ssl_certfile, "ssl_keyfile": args.ssl_keyfile} if ssl else {}
    if not ssl:
        print("note: no TLS certificate; https viewers (neuroglancer-demo, vizarr) cannot fetch from an http server", file=sys.stderr)
    uvicorn.run(create_app(config), host="0.0.0.0", port=port, log_level="warning", **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
