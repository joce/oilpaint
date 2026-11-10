"""Small image filters (numpy only)."""

import numpy as np


def _box_blur(a: np.ndarray, r: int, axis: int | None = None) -> np.ndarray:
    """Box blur of radius r, the edges repeating the border value: along
    ``axis``, or (by default) along the first two axes in turn, in float64
    from first to last. The result has ``a``'s type."""
    if r <= 0:
        return a
    if axis is not None:
        return _box_pass(a, r, axis).astype(a.dtype)
    out = a.astype(np.float64)
    for ax in (0, 1):
        out = _box_pass(out, r, ax)
    return out.astype(a.dtype)


def _box_pass(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """One pass of :func:`_box_blur` along ``axis`` (float64: running sums)."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


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


def _smoothstep(e0, e1, x):
    """0 below ``e0``, 1 beyond ``e1`` (either way round), and a smooth S
    between them."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear(grid: np.ndarray, y, x):
    """Sample ``grid`` (H, W, ...) at the fractional indices ``y`` and ``x``
    (arrays, 0..H-1 and 0..W-1), linearly between the four cells around
    each point."""
    H, W = grid.shape[:2]
    y0, x0 = np.minimum(y.astype(int), H - 2), np.minimum(x.astype(int), W - 2)
    fy, fx = y - y0, x - x0
    if grid.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy) + (
        grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx
    ) * fy
