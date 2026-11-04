"""What a brush prints when it is pressed straight down (a dab).

The pattern is a height map of the brush's bristle tips on a square grid
spanning -SPAN..SPAN brush half-widths: clumps of bristle ends of slightly
different lengths, each a little cone, 1 where it touches first and 0 where
it never touches. The engine compares it with a threshold that falls as the
pressure rises (see ``footprint``): a light tap prints only the longest tips,
sparse dots; a firmer one spreads them into bigger dots until the mark closes
up.

A brush keeps its layout (it comes from the brush type), turned to a new
angle on every tap, as a painter's hand turns.
"""

from __future__ import annotations

import functools
import zlib

import numpy as np

SPAN = 1.3


def grid_size(r_px: float) -> int:
    """Grid cells across the pattern: about two per pixel of the brush."""
    return int(np.clip(np.ceil(2 * SPAN * r_px / 0.5), 96, 320))


def _coords(n: int):
    t = (np.arange(n) + 0.5) / n * 2 * SPAN - SPAN
    return np.meshgrid(t, t)  # (across the brush, along the stroke)


def _bristle_tips(n, spacing, rng):
    """Cones at jittered hexagonal positions ``spacing`` apart (half-widths),
    of random heights: the longest bristle clumps touch first."""
    X, Y = _coords(n)
    rows = np.arange(-1.2, 1.2 + spacing, 0.866 * spacing)
    pts = [
        (x, y)
        for j, y in enumerate(rows)
        for x in np.arange(
            -1.2 + (0.5 * spacing if j % 2 else 0.0), 1.2 + spacing, spacing
        )
    ]
    pts = np.asarray(pts) + rng.uniform(-0.35, 0.35, (len(pts), 2)) * spacing
    heights = rng.uniform(0.25, 1.0, len(pts)) ** 0.6
    radii = 0.62 * spacing * rng.uniform(0.75, 1.25, len(pts))
    out = np.zeros((n, n))
    cell = 2 * SPAN / n
    for (px, py), h, rad in zip(pts, heights, radii, strict=True):
        x0 = max(int((px - rad + SPAN) / cell), 0)
        x1 = min(int((px + rad + SPAN) / cell) + 2, n)
        y0 = max(int((py - rad + SPAN) / cell), 0)
        y1 = min(int((py + rad + SPAN) / cell) + 2, n)
        if x0 >= x1 or y0 >= y1:
            continue
        d = np.hypot(X[y0:y1, x0:x1] - px, Y[y0:y1, x0:x1] - py)
        cone = h * np.clip(1.0 - d / rad, 0.0, 1.0) ** 0.6
        np.maximum(out[y0:y1, x0:x1], cone, out=out[y0:y1, x0:x1])
    return out


@functools.lru_cache(maxsize=64)
def _layout(seed: int, r_px: float, tips: int, bristles: float):
    n = grid_size(r_px)
    rng = np.random.default_rng(seed)
    # tips about 1.5 px apart on fine hair to 4 px on coarse hog, or as many as
    # asked for
    spacing_px = 1.5 + 2.5 * (1.0 - bristles)
    spacing = float(np.sqrt(np.pi / tips)) if tips > 0 else spacing_px / r_px
    out = np.clip(_bristle_tips(n, min(spacing, 1.0), rng), 0.0, 1.0)
    out.setflags(write=False)
    return out


def dab_pattern(bt, r_px: float, rng: np.random.Generator) -> np.ndarray:
    """The face of brush type ``bt`` (half-width ``r_px``), turned to a random
    angle for this dab."""
    seed = zlib.crc32(repr(bt).encode())
    base = _layout(seed, round(2 * r_px) / 2, int(bt.tips), float(bt.bristles))
    n = base.shape[0]
    X, Y = _coords(n)
    a = rng.uniform(0.0, 2 * np.pi)
    ca, sa = np.cos(a), np.sin(a)
    xr, yr = ca * X + sa * Y, -sa * X + ca * Y
    ix = ((xr + SPAN) / (2 * SPAN) * n).astype(int)
    iy = ((yr + SPAN) / (2 * SPAN) * n).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    out = np.zeros((n, n))
    out[ok] = base[iy[ok], ix[ok]]
    return out
