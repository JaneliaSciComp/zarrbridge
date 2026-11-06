"""Downsample factor policy and level layout (spec §9.2, §9.3)."""
from __future__ import annotations

import fnmatch
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zarrbridge.readers.base import ImageInfo
from zarrbridge.store.omezarr import level_transforms

TILE = 512
SPATIAL = ("z", "y", "x")


@dataclass(frozen=True)
class Level:
    index: int
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    factor: tuple[int, ...]  # cumulative vs level 0
    scale: list[float]
    translation: list[float] | None


def _axis_index(image: ImageInfo) -> dict[str, int]:
    return {a.rstrip("0123456789"): i for i, a in enumerate(image.axis_names)}


def parse_factor(text: str, image: ImageInfo) -> tuple[int, ...]:
    """'2,2,1' is (x, y, z) like mrc-ng-server's scale keys; non-spatial axes get 1."""
    vals = [int(v) for v in text.replace("_", ",").split(",") if v.strip()]
    ax = _axis_index(image)
    f = [1] * len(image.shape)
    for name, v in zip(("x", "y", "z"), vals):
        if name in ax:
            f[ax[name]] = v
    return tuple(f)


def choose_factor(image: ImageInfo, config: Any, path: Path | str) -> tuple[int, ...]:
    name = os.path.basename(str(path))
    for glob, text in config.build.factor_by_glob.items():
        if fnmatch.fnmatch(str(path), glob) or fnmatch.fnmatch(name, glob):
            return parse_factor(text, image)
    if config.build.factor != "auto":
        return parse_factor(config.build.factor, image)
    ax = _axis_index(image)
    if "z" not in ax or image.shape[ax["z"]] == 1:
        return parse_factor("2,2", image)
    z = ax["z"]
    zc = image.native_chunks[z]
    spacing = image.scale
    xy = [spacing[ax[a]] for a in ("y", "x") if a in ax]
    zs = spacing[z]
    if zc == 1 or image.is_stack or (xy and zs > 2 * max(xy)):
        return parse_factor("2,2,1", image)
    if xy and all(abs(zs - s) <= 0.1 * max(zs, s) for s in xy):
        return parse_factor("2,2,2", image)
    return parse_factor("2,2,1", image)


def plan_levels(image: ImageInfo, factor: tuple[int, ...], min_axis: int = 64, max_levels: int = 8) -> list[Level]:
    shape = tuple(int(s) for s in image.shape)
    native = tuple(int(c) for c in image.native_chunks)
    ax = _axis_index(image)
    spatial_idx = [ax[a] for a in ("y", "x") if a in ax]
    # plane-slab native chunks (strips / full planes) -> 512 tiles in the two trailing spatial axes
    slab = any(native[i] == shape[i] for i in spatial_idx[-1:]) and len(spatial_idx) == 2
    levels = [Level(0, shape, native, tuple(1 for _ in shape), list(image.scale), list(image.translation) if image.translation else None)]
    while len(levels) < max_levels:
        prev = levels[-1]
        # All downsampled axes halve together (levels stay isotropic in the downsampled axes; Neuroglancer's
        # level selection breaks on a level that is anisotropic relative to the one below it). Stop when every
        # downsampled axis would fall below min_axis (spec §9.2).
        step = tuple(f if f > 1 else 1 for f in factor)
        if all(s == 1 for s in step) or all(math.ceil(n / st) < min_axis for n, st in zip(prev.shape, step) if st > 1):
            break
        new_factor = tuple(pf * s for pf, s in zip(prev.factor, step))
        new_shape = tuple(math.ceil(s / st) for s, st in zip(prev.shape, step))
        chunks = []
        for i, (n, c) in enumerate(zip(new_shape, native)):
            if slab and i in spatial_idx and factor[i] > 1:
                chunks.append(min(TILE, n))
            else:
                chunks.append(min(c, n))
        scale, translation = level_transforms(image.scale, new_factor, image.translation)
        levels.append(Level(len(levels), new_shape, tuple(chunks), new_factor, scale, translation))
    return levels


def plan_dict(levels: list[Level], factor: tuple[int, ...], codec: str = "zstd:3", method: str = "mean") -> dict:
    return {"factor": list(factor), "levels": len(levels), "level_chunks": [list(l.chunks) for l in levels],
            "level_shapes": [list(l.shape) for l in levels], "codec": codec, "method": method}
