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
import math
import zlib

import numpy as np
from numba import njit

SPAN = 1.3


def grid_size(r_px: float) -> int:
    """Grid cells across the pattern: about two per pixel of the brush."""
    return int(np.clip(np.ceil(2 * SPAN * r_px / 0.5), 96, 320))


def _ticks(n: int) -> np.ndarray:
    """Where the grid's cells are, across (and along) the brush."""
    return (np.arange(n) + 0.5) / n * 2 * SPAN - SPAN


def _coords(n: int):
    t = _ticks(n)
    return np.meshgrid(t, t)  # (across the brush, along the stroke)


@njit(cache=True, error_model="numpy")
def _cone_bases(t, cell, pts, radii):
    """The cells under each cone (flat indices into the n x n grid, cone after
    cone), and for each 1 - (its distance from the cone's apex) / (the cone's
    radius), clipped to 0..1."""
    n = t.shape[0]
    count = 0
    for k in range(pts.shape[0]):
        px, py, rad = pts[k, 0], pts[k, 1], radii[k]
        x0 = max(int((px - rad + SPAN) / cell), 0)
        x1 = min(int((px + rad + SPAN) / cell) + 2, n)
        y0 = max(int((py - rad + SPAN) / cell), 0)
        y1 = min(int((py + rad + SPAN) / cell) + 2, n)
        if x0 < x1 and y0 < y1:
            count += (x1 - x0) * (y1 - y0)
    index = np.empty(count, np.int64)
    base = np.empty(count)
    cone = np.empty(count, np.int64)
    q = 0
    for k in range(pts.shape[0]):
        px, py, rad = pts[k, 0], pts[k, 1], radii[k]
        x0 = max(int((px - rad + SPAN) / cell), 0)
        x1 = min(int((px + rad + SPAN) / cell) + 2, n)
        y0 = max(int((py - rad + SPAN) / cell), 0)
        y1 = min(int((py + rad + SPAN) / cell) + 2, n)
        if x0 >= x1 or y0 >= y1:
            continue
        for j in range(y0, y1):
            for i in range(x0, x1):
                d = math.hypot(t[i] - px, t[j] - py)
                index[q] = j * n + i
                base[q] = min(max(1.0 - d / rad, 0.0), 1.0)
                cone[q] = k
                q += 1
    return index, base, cone


@njit(cache=True, error_model="numpy")
def _highest(n, index, heights):
    """The highest of the heights landing on each cell of an n x n grid (0
    where none does)."""
    out = np.zeros(n * n)
    for q in range(index.shape[0]):
        if heights[q] > out[index[q]]:
            out[index[q]] = heights[q]
    return out.reshape(n, n)


@njit(cache=True, error_model="numpy")
def _turned(base, t, ca, sa):
    """The pattern ``base`` (n x n, on the grid ``t``) turned by the angle
    whose cosine and sine are ``ca``, ``sa``."""
    n = base.shape[0]
    out = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            xr = ca * t[i] + sa * t[j]
            yr = -sa * t[i] + ca * t[j]
            ix = int((xr + SPAN) / (2 * SPAN) * n)
            iy = int((yr + SPAN) / (2 * SPAN) * n)
            if 0 <= ix < n and 0 <= iy < n:
                out[j, i] = base[iy, ix]
    return out


def _bristle_tips(n, spacing, rng):
    """Cones at jittered hexagonal positions ``spacing`` apart (half-widths),
    of random heights: the longest bristle clumps touch first."""
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
    # (the powers in numpy, as numpy works them out)
    index, base, cone = _cone_bases(_ticks(n), 2 * SPAN / n, pts, radii)
    return _highest(n, index, heights[cone] * base**0.6)


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
    a = rng.uniform(0.0, 2 * np.pi)
    ca, sa = np.cos(a), np.sin(a)
    return _turned(base, _ticks(n), float(ca), float(sa))
