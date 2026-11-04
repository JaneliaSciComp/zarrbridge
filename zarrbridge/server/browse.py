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


# Neuroglancer reads OME `multiscales` but applies `omero` colours/windows only through its "multi-channel"
# UI action. These links build the same state that action produces: one additive layer per channel, coloured
# and windowed from `omero.channels` (R, G, B, white defaults), so colour images open in colour.
_CHANNEL_SHADER = """#uicontrol invlerp contrast
#uicontrol vec3 color color
void main() {
  float contrast_value = contrast();
  if (VOLUME_RENDERING) {
    emitRGBA(vec4(color * contrast_value, contrast_value));
  }
  else {
    emitRGB(color * contrast_value);
  }
}
"""
_DEFAULT_COLORS = ["#ff0000", "#00ff00", "#0000ff", "#ffffff"]


def neuroglancer_state(url: str, name: str, group_meta: dict | None = None) -> dict:
    layers: list[dict] = []
    ms = (group_meta or {}).get("attributes", {}).get("ome", {}).get("multiscales", [{}])[0]
    axes = [a["name"] for a in ms.get("axes", [])]
    channels = (group_meta or {}).get("attributes", {}).get("ome", {}).get("omero", {}).get("channels") or []
    n_channels = len(channels) if channels else ((group_meta or {}).get("zarrbridge_shape") or {}).get("c", 0)
    if "c" in axes and n_channels >= 2:
        for i in range(n_channels):
            ch = channels[i] if i < len(channels) else {}
            color = ch.get("color") or _DEFAULT_COLORS[i % 4].lstrip("#")
            win = ch.get("window") or {}
            controls: dict = {"color": f"#{color.lstrip('#')}"}
            if {"start", "end"} <= win.keys():
                contrast: dict = {"range": [win["start"], win["end"]]}
                if {"min", "max"} <= win.keys():
                    contrast["window"] = [win["min"], win["max"]]
                controls["contrast"] = contrast
            layer = {"type": "image", "source": f"zarr3://{url}", "name": ch.get("label") or f"{name} ch{i}",
                     "localDimensions": {"c'": [1, ""]}, "localPosition": [i], "shader": _CHANNEL_SHADER,
                     "shaderControls": controls, "blend": "additive", "opacity": 1, "tab": "rendering"}
            if not ch.get("active", i <= 3):
                layer["archived"] = True
            layers.append(layer)
    else:
        layers.append({"type": "image", "source": f"zarr3://{url}", "name": name, "tab": "rendering"})
    return {"layers": layers, "selectedLayer": {"visible": True, "layer": layers[0]["name"]}}


def neuroglancer_link(url: str, name: str, viewer_base: str | None = None, group_meta: dict | None = None) -> str:
    """`viewer_base` is this server's own /neuroglancer/ when a client build is configured: same origin, so
    browsers need neither CORS nor a Local Network Access permission. `group_meta` (the image's zarr.json)
    enables per-channel colour layers."""
    state = neuroglancer_state(url, name, group_meta)
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
