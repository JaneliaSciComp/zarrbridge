import math

import numpy as np

from zarrbridge.build.downsample import block_mean


def _reference_block_mean(arr, factors):
    out_shape = tuple(math.ceil(s / f) for s, f in zip(arr.shape, factors))
    out = np.empty(out_shape, dtype=arr.dtype)
    for idx in np.ndindex(*out_shape):
        block = arr[tuple(slice(i * f, (i + 1) * f) for i, f in zip(idx, factors))]
        m = float(block.astype(np.float64).mean())
        out[idx] = math.floor(m + 0.5) if m >= 0 else math.ceil(m - 0.5)
    return out


def test_multi_axis_matches_independent_reference():
    rng = np.random.default_rng(0)
    arr = rng.integers(-3000, 3000, size=(8, 8, 8)).astype(np.int16)
    np.testing.assert_array_equal(block_mean(arr, (2, 2, 2)), _reference_block_mean(arr, (2, 2, 2)))
    arr = rng.integers(0, 2000, size=(5, 7, 3)).astype(np.int16)
    np.testing.assert_array_equal(block_mean(arr, (2, 2, 2)), _reference_block_mean(arr, (2, 2, 2)))
    arr4 = rng.integers(0, 60000, size=(2, 3, 7, 9)).astype(np.uint16)
    np.testing.assert_array_equal(block_mean(arr4, (1, 1, 2, 2)), _reference_block_mean(arr4, (1, 1, 2, 2)))


def test_edge_cases():
    assert block_mean(np.arange(4, dtype=np.int16).reshape(1, 1, 4), (1, 1, 2)).tolist() == [[[1, 3]]]
    assert block_mean(np.array([10, 20, 30, 40, 50], dtype=np.int16).reshape(1, 1, 5), (1, 1, 2)).tolist() == [[[15, 35, 50]]]
    assert block_mean(np.full((2, 2, 2), 32000, dtype=np.int16), (2, 2, 2)).item() == 32000
    assert block_mean(np.array([-1, -2], dtype=np.int16).reshape(1, 1, 2), (1, 1, 2)).item() == -2
    r = block_mean(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 1, 4), (1, 1, 2))
    np.testing.assert_allclose(r, [[[1.5, 3.5]]]) and r.dtype == np.float32
    assert block_mean(np.full((2, 2), 2**31 - 1, dtype=np.int32), (2, 2)).item() == 2**31 - 1
