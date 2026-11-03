"""Small image filters (numpy only)."""

import numpy as np


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur with edge clamping (radius r)."""
    if r <= 0:
        return a
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out.astype(a.dtype)


def _max_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Maximum over a (2r+1)-square neighbourhood (separable, edges clamped)."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        res = np.take(p, np.arange(0, n), axis=axis)
        for k in range(1, 2 * r + 1):
            np.maximum(res, np.take(p, np.arange(k, k + n), axis=axis), out=res)
        out = res
    return out


def _smooth121(a: np.ndarray) -> np.ndarray:
    """Light [1, 2, 1] smoothing in both directions (softens pixel steps, keeps
    fine grooves)."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        p = np.pad(
            out, [(1, 1) if k == axis else (0, 0) for k in range(2)], mode="edge"
        )
        n = out.shape[axis]
        out = (
            0.25 * np.take(p, np.arange(0, n), axis=axis)
            + 0.5 * out
            + 0.25 * np.take(p, np.arange(2, n + 2), axis=axis)
        )
    return out
