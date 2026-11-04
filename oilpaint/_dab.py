"""What a tool's face prints when it is pressed straight down (a dab).

Each pattern is a height map of the tool's face on a square grid spanning
-SPAN..SPAN brush half-widths: 1 where it touches first, 0 where it never
touches. The engine compares it with a threshold that falls as the pressure
rises (see ``footprint``): a light tap prints only the highest points, a firm
one more of the face.

* bristles: separate tips, clumps of bristle ends of slightly different
  lengths, each a little cone (a harder tap spreads it into a bigger dot);
* sponge: an irregular web of walls around holes, the surface uneven, the
  outline ragged; the holes stay open however hard it is pressed.

A brush keeps its layout (it comes from the brush type), turned to a new
angle on every tap, as a painter's hand turns.
"""

from __future__ import annotations

import functools
import zlib

import numpy as np

from .noise import fractal_noise_2d

SPAN = 1.3


def grid_size(r_px: float) -> int:
    """Grid cells across the pattern: about two per pixel of the brush."""
    return int(np.clip(np.ceil(2 * SPAN * r_px / 0.5), 96, 320))


def _coords(n: int):
    t = (np.arange(n) + 0.5) / n * 2 * SPAN - SPAN
    return np.meshgrid(t, t)  # (across the brush, along the stroke)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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


def _cells(X, Y, spacing, rng, wall):
    """1 on the walls between irregular holes about ``spacing`` apart, 0 in
    the holes; ``wall`` sets how thick the walls are."""
    m = int(np.ceil(2 * (SPAN + 0.3) / spacing)) + 1
    base = -SPAN - 0.3 + spacing * np.arange(m)
    gx, gy = np.meshgrid(base, base)
    seeds = np.stack([gx, gy], -1) + rng.uniform(-0.45, 0.45, (m, m, 2)) * spacing
    weight = rng.uniform(0.7, 1.35, (m, m))  # some holes bigger than others
    ix = np.clip(((X + SPAN + 0.3) / spacing).astype(int), 0, m - 1)
    iy = np.clip(((Y + SPAN + 0.3) / spacing).astype(int), 0, m - 1)
    f1 = np.full(X.shape, np.inf)
    f2 = np.full(X.shape, np.inf)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            jx, jy = np.clip(ix + dx, 0, m - 1), np.clip(iy + dy, 0, m - 1)
            s = seeds[jy, jx]
            d = np.hypot(X - s[..., 0], Y - s[..., 1]) / weight[jy, jx]
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return 1.0 - _smoothstep(0.0, wall * spacing, f2 - f1)


def _sponge(n, spacing, rng):
    """A sea sponge's face: big holes and small pores in an uneven web, some
    parts standing proud (they print first), the outline ragged."""
    X, Y = _coords(n)
    web = _cells(X, Y, spacing, rng, 0.75) * (
        0.35 + 0.65 * _cells(X, Y, 0.4 * spacing, rng, 0.9)
    )
    rough = fractal_noise_2d(n, n, n / 6.0, rng, octaves=3)
    rough = (rough - rough.min()) / max(np.ptp(rough), 1e-9)
    face = web * (0.2 + 0.8 * rough)
    # a ragged outline
    ang = np.arctan2(Y, X)
    edge = (
        1.0
        + 0.08 * np.sin(3 * ang + rng.uniform(0, 6))
        + 0.06 * np.sin(5 * ang + rng.uniform(0, 6))
    )
    return face * _smoothstep(edge, edge - 0.18, np.hypot(X, Y))


@functools.lru_cache(maxsize=64)
def _layout(pattern: str, seed: int, r_px: float, tips: int, bristles: float):
    n = grid_size(r_px)
    rng = np.random.default_rng(seed)
    if pattern == "sponge":
        # holes about a fifth of the sponge across, or as many as asked for
        spacing = float(np.sqrt(np.pi / tips)) if tips > 0 else 0.22
        out = _sponge(n, spacing, rng)
    else:
        # tips about 1.5 px apart on fine hair to 4 px on coarse hog, or as many
        # as asked for
        spacing_px = 1.5 + 2.5 * (1.0 - bristles)
        spacing = float(np.sqrt(np.pi / tips)) if tips > 0 else spacing_px / r_px
        out = _bristle_tips(n, min(spacing, 1.0), rng)
    out = np.clip(out, 0.0, 1.0)
    out.setflags(write=False)
    return out


def dab_pattern(bt, r_px: float, rng: np.random.Generator) -> np.ndarray:
    """The face of brush type ``bt`` (half-width ``r_px``), turned to a random
    angle for this dab."""
    seed = zlib.crc32(repr(bt).encode())
    base = _layout(
        bt.pattern, seed, round(2 * r_px) / 2, int(bt.tips), float(bt.bristles)
    )
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
