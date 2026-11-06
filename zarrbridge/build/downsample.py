"""Block-mean downsampling (port of mrc-ng-server downsample.py, generalised to N axes).
factors are in the same axis order as the array shape."""
from __future__ import annotations

import numpy as np


def _sum_and_count(arr: np.ndarray, factor: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[axis]
    indices = np.arange(0, n, factor)
    summed = np.add.reduceat(arr, indices, axis=axis)
    counts = np.diff(np.append(indices, n))
    shape = [1] * arr.ndim
    shape[axis] = len(counts)
    return summed, counts.reshape(shape)


def block_mean(arr: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Mean of each block, as one division at the end (no per-axis intermediate rounding).
    Integers round half away from zero and clip to the dtype; the trailing short block divides by its real count."""
    src_dtype = arr.dtype
    floating = np.issubdtype(src_dtype, np.floating)
    # ponytail: int32 accumulator holds int16 inputs up to prod(factors) ~ 65000; per-step factors are 1 or 2.
    # Widen to int64 for 32-bit integer inputs.
    if floating:
        acc = arr.astype(np.float64)
    elif src_dtype.itemsize >= 4:
        acc = arr.astype(np.int64)
    else:
        acc = arr.astype(np.int32)
    divisor: np.ndarray | int = 1
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        acc, counts = _sum_and_count(acc, factor, axis)
        divisor = divisor * counts
    result = acc / divisor
    if floating:
        return result.astype(src_dtype)
    rounded = np.where(result >= 0, np.floor(result + 0.5), np.ceil(result - 0.5))
    info = np.iinfo(src_dtype)
    return np.clip(rounded, info.min, info.max).astype(src_dtype)
