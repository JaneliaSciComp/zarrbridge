"""OME-Zarr 0.5 / zarr v3 metadata builders shared by readers and the builder."""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

ZARR_KEY_ENCODING = {"name": "default", "configuration": {"separator": "/"}}
BYTES_LE = [{"name": "bytes", "configuration": {"endian": "little"}}]


def array_meta(shape: Sequence[int], chunk_shape: Sequence[int], dtype: Any, codecs: list[dict] | None = None,
               dimension_names: Sequence[str] | None = None, fill_value: Any = 0, attributes: dict | None = None) -> dict:
    dt = np.dtype(dtype).newbyteorder("=")
    meta = {
        "zarr_format": 3, "node_type": "array",
        "shape": [int(s) for s in shape],
        "data_type": dt.name,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [int(c) for c in chunk_shape]}},
        "chunk_key_encoding": ZARR_KEY_ENCODING,
        "fill_value": fill_value,
        "codecs": codecs if codecs is not None else BYTES_LE,
        "attributes": attributes or {},
    }
    if dimension_names is not None:
        meta["dimension_names"] = list(dimension_names)
    return meta


def axis(name: str, unit: str | None = None) -> dict:
    t = {"t": "time", "c": "channel"}.get(name, "space")
    a: dict[str, Any] = {"name": name, "type": t}
    if unit and t != "channel":
        a["unit"] = unit
    return a


def level_transforms(scale0: Sequence[float], factor: Sequence[int],
                     translation0: Sequence[float] | None = None) -> tuple[list[float], list[float] | None]:
    """scale = s0*f ; translation = s0*(f-1)/2 (pixel-centre convention). None translation at factor 1."""
    scale = [float(s) * int(f) for s, f in zip(scale0, factor)]
    base = list(translation0) if translation0 is not None else [0.0] * len(scale)
    if all(int(f) == 1 for f in factor) and translation0 is None:
        return scale, None
    return scale, [float(b) + float(s) * (int(f) - 1) / 2 for b, s, f in zip(base, scale0, factor)]


def dataset_entry(path: str, scale: Sequence[float], translation: Sequence[float] | None) -> dict:
    ct: list[dict] = [{"type": "scale", "scale": [float(s) for s in scale]}]
    if translation is not None:
        ct.append({"type": "translation", "translation": [float(t) for t in translation]})
    return {"path": path, "coordinateTransformations": ct}


def multiscales_group(axes: list[dict], datasets: list[dict], name: str,
                      extra_ome: dict | None = None, extra_attrs: dict | None = None) -> dict:
    ome: dict[str, Any] = {"version": "0.5", "multiscales": [{"name": name, "axes": axes, "datasets": datasets}]}
    if extra_ome:
        ome.update(extra_ome)
    attrs: dict[str, Any] = {"ome": ome}
    if extra_attrs:
        attrs.update(extra_attrs)
    return {"zarr_format": 3, "node_type": "group", "attributes": attrs}


def plain_group(attributes: dict | None = None) -> dict:
    return {"zarr_format": 3, "node_type": "group", "attributes": attributes or {}}


def dumps(meta: dict) -> bytes:
    return json.dumps(meta, indent=2).encode()


def grid_shape(shape: Sequence[int], chunks: Sequence[int]) -> tuple[int, ...]:
    return tuple(-(-int(s) // int(c)) for s, c in zip(shape, chunks))
