"""Axes, pixel sizes and units from tifffile series metadata."""
from __future__ import annotations

from typing import Any

import tifffile

AXIS_MAP = {"T": "t", "Z": "z", "C": "c", "Y": "y", "X": "x", "S": "c", "I": "z", "Q": "z"}
UNIT_MAP = {"µm": "micrometer", "um": "micrometer", "micron": "micrometer", "microns": "micrometer",
            "nm": "nanometer", "mm": "millimeter", "cm": "centimeter", "m": "meter", "Å": "angstrom",
            "angstrom": "angstrom", "pixel": None, "": None, "inch": "inch"}


def axis_names(tiff_axes: str) -> list[str]:
    names = [AXIS_MAP.get(a, a.lower()) for a in tiff_axes]
    seen: set[str] = set()
    out = []
    for n in names:  # duplicates (e.g. C and S) get a numeric suffix so dimension_names stay unique
        k = n
        i = 1
        while k in seen:
            i += 1
            k = f"{n}{i}"
        seen.add(k)
        out.append(k)
    return out


def physical_sizes(tf: tifffile.TiffFile, series_index: int, page: tifffile.TiffPage) -> tuple[dict[str, float], str | None]:
    """{'x': sx, 'y': sy, 'z': sz, 't': dt} and the unit (OME unit name) when known."""
    sizes: dict[str, float] = {}
    unit: str | None = None
    if tf.is_ome and tf.ome_metadata:
        try:
            md = tifffile.xml2dict(tf.ome_metadata)["OME"]["Image"]
            md = md if isinstance(md, list) else [md]
            px = md[min(series_index, len(md) - 1)]["Pixels"]
            for k, ax in (("PhysicalSizeX", "x"), ("PhysicalSizeY", "y"), ("PhysicalSizeZ", "z"), ("TimeIncrement", "t")):
                if k in px:
                    sizes[ax] = float(px[k])
            u = px.get("PhysicalSizeXUnit") or px.get("PhysicalSizeYUnit") or px.get("PhysicalSizeZUnit")
            unit = UNIT_MAP.get(u, u) if u is not None else ("micrometer" if sizes else None)
            return sizes, unit
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    ij = tf.imagej_metadata if tf.is_imagej else None
    res = page.resolution if page is not None else (0, 0)
    if ij:
        if ij.get("spacing"):
            sizes["z"] = float(ij["spacing"])
        if ij.get("finterval"):
            sizes["t"] = float(ij["finterval"])
        unit = UNIT_MAP.get(ij.get("unit", ""), ij.get("unit"))
        if res and res[0] and res[1]:
            sizes["x"], sizes["y"] = 1.0 / float(res[0]), 1.0 / float(res[1])
        return sizes, unit
    if res and res[0] and res[1] and page.resolutionunit in (2, 3):
        per_unit_um = 25400.0 if page.resolutionunit == 2 else 10000.0
        sizes["x"], sizes["y"] = per_unit_um / float(res[0]), per_unit_um / float(res[1])
        return sizes, "micrometer"
    return sizes, None


def scale_for(axes: list[str], sizes: dict[str, float]) -> list[float]:
    return [float(sizes.get(a, 1.0)) for a in axes]


def ome_axes(axes: list[str], unit: str | None, sizes: dict[str, float]) -> list[dict]:
    from zarrbridge.store.omezarr import axis
    out = []
    for a in axes:
        base = a.rstrip("0123456789")
        if base == "t":
            out.append(axis("t", "second" if "t" in sizes else None))
        elif base == "c":
            out.append({"name": a, "type": "channel"})
        else:
            d = axis(base if base in ("z", "y", "x") else a, unit)
            d["name"] = a
            out.append(d)
    return out


_RANK = {"t": 0, "c": 1, "z": 2, "y": 4, "x": 5}


def canonical_order(axes: list[str]) -> tuple[int, ...]:
    """Transpose order putting axes in NGFF order (t, c, z, y, x; unknown axes before z):
    encoded (file) axis i corresponds to presented axis order[i]."""
    def rank(name: str) -> float:
        return _RANK.get(name.rstrip("0123456789"), 3)
    presented = sorted(range(len(axes)), key=lambda i: (rank(axes[i]), i))  # presented position -> file axis
    order = [0] * len(axes)
    for pos, file_axis in enumerate(presented):
        order[file_axis] = pos
    return tuple(order)


def inverse(order: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(order)
    for i, o in enumerate(order):
        inv[o] = i
    return tuple(inv)
