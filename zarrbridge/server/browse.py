"""Directory listing with viewer links (spec §11.1)."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from zarrbridge.readers.registry import sniffable

NEUROGLANCER = "https://neuroglancer-demo.appspot.com/#!"
VIZARR = "https://hms-dbmi.github.io/vizarr/?source="
MAX_ENTRIES = 500


def neuroglancer_link(url: str, name: str, viewer_base: str | None = None) -> str:
    """`viewer_base` is this server's own /neuroglancer/ when the client is bundled: same origin, so browsers
    need neither CORS nor a Local Network Access permission."""
    # side panel open on the layer's rendering tab, so the contrast window is one click away
    state = {"layers": [{"type": "image", "source": f"zarr3://{url}", "name": name, "tab": "rendering"}],
             "selectedLayer": {"visible": True, "layer": name}}
    base = f"{viewer_base.rstrip('/')}/#!" if viewer_base else NEUROGLANCER
    return base + quote(json.dumps(state, separators=(",", ":")), safe="")


def render_roots(roots: dict[str, str]) -> str:
    items = "".join(f'<li><a href="/{quote(n)}/">{html.escape(n)}/</a></li>' for n in sorted(roots))
    return _page("zarrbridge", f"<h1>zarrbridge</h1><ul>{items}</ul>")


def render_listing(root: str, relpath: str, directory: Path, base_url: str, viewer_base: str | None = None) -> str:
    entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())[:MAX_ENTRIES]
    here = f"/{root}/{relpath}".rstrip("/")
    parts = [f"<h1>{html.escape(here)}/</h1>"]
    parent = here.rsplit("/", 1)[0] or "/"
    parts.append(f'<p><a href="{html.escape(parent)}/">.. (parent)</a></p><ul>')
    for e in entries:
        if e.name.startswith("."):
            continue
        href = f"{here}/{quote(e.name)}"
        if e.is_dir():
            parts.append(f'<li><a href="{href}/">{html.escape(e.name)}/</a></li>')
        elif sniffable(e):
            url = f"{base_url}{href}"
            parts.append(
                f'<li>{html.escape(e.name)} &mdash; '
                f'<a href="{html.escape(neuroglancer_link(url, e.name, viewer_base))}" target="_blank">Neuroglancer</a> &middot; '
                f'<a href="{html.escape(VIZARR + quote(url, safe=""))}" target="_blank">vizarr</a> &middot; '
                f'<a href="/_status{href}">status</a> &middot; <a href="{href}/zarr.json">zarr.json</a></li>')
    parts.append("</ul>")
    return _page(here, "".join(parts))


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:2em}li{margin:.25em 0}</style></head>"
            f"<body>{body}</body></html>")
