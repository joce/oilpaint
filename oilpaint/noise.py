"""Small procedural noise helpers (numpy only)."""

from __future__ import annotations

import numpy as np


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def value_noise_2d(h: int, w: int, cell: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise in [0, 1] with features about ``cell`` pixels wide."""
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    grid = rng.random((gh, gw))
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    yi, xi = y.astype(int), x.astype(int)
    fy, fx = _smooth(y - yi)[:, None], _smooth(x - xi)[None, :]
    g00 = grid[yi][:, xi]
    g01 = grid[yi][:, xi + 1]
    g10 = grid[yi + 1][:, xi]
    g11 = grid[yi + 1][:, xi + 1]
    top = g00 + (g01 - g00) * fx
    bot = g10 + (g11 - g10) * fx
    return top + (bot - top) * fy


def fractal_noise_2d(h, w, cell, rng, octaves=4, gain=0.5) -> np.ndarray:
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        out += amp * value_noise_2d(h, w, max(cell, 1.0), rng)
        total += amp
        amp *= gain
        cell /= 2.0
    return out / total


def value_noise_1d(n: int, cell: float, rng: np.random.Generator) -> np.ndarray:
    grid = rng.random(int(n / cell) + 3)
    x = np.arange(n) / cell
    xi = x.astype(int)
    f = _smooth(x - xi)
    return grid[xi] + (grid[xi + 1] - grid[xi]) * f


def canvas_weave(h: int, w: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """Height field in [0, 1] resembling primed plain-weave canvas."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    jitter = fractal_noise_2d(h, w, period * 6, rng, octaves=2) - 0.5
    xs = x + jitter * period * 0.6
    ys = y - jitter * period * 0.6
    cell_x = np.floor(xs / period)
    cell_y = np.floor(ys / period)
    fx = np.sin(np.pi * xs / period) ** 2
    fy = np.sin(np.pi * ys / period) ** 2
    over = ((cell_x + cell_y) % 2) == 0
    weave = np.where(over, fx * (0.6 + 0.4 * fy), fy * (0.6 + 0.4 * fx))
    grain = fractal_noise_2d(h, w, period * 1.5, rng, octaves=3)
    out = 0.75 * weave + 0.25 * grain
    out -= out.min()
    return out / max(out.max(), 1e-9)


# ---------------------------------------------------------------------------
# Surface textures (height fields in 0..1)
# ---------------------------------------------------------------------------


def lattice_noise(U, V, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise evaluated at arbitrary lattice coordinates (U, V)."""
    U = U - U.min()
    V = V - V.min()
    grid = rng.random((int(U.max()) + 3, int(V.max()) + 3))
    iu, iv = U.astype(int), V.astype(int)
    fu, fv = _smooth(U - iu), _smooth(V - iv)
    g00, g01 = grid[iu, iv], grid[iu, iv + 1]
    g10, g11 = grid[iu + 1, iv], grid[iu + 1, iv + 1]
    a = g00 + (g01 - g00) * fv
    b = g10 + (g11 - g10) * fv
    return a + (b - a) * fu


def aniso_noise(h, w, along, across, angle_deg, rng, octaves=2) -> np.ndarray:
    """Noise stretched along a direction: features ``along`` px long and
    ``across`` px wide, oriented ``angle_deg`` from horizontal."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.radians(angle_deg)
    u = x * np.cos(a) + y * np.sin(a)
    v = -x * np.sin(a) + y * np.cos(a)
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        out += amp * lattice_noise(u / max(along, 1.0), v / max(across, 0.5), rng)
        total += amp
        amp *= 0.5
        along /= 2.0
        across /= 2.0
    return out / total


def _normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def woven(h, w, period, rng, irregularity=0.3, twill=False) -> np.ndarray:
    """Woven cloth: threads going over and under each other.

    ``irregularity`` 0 gives machine-even cotton; higher values give linen,
    with wandering threads of uneven thickness and slubs (thick spots).
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    wander = 0.2 + 0.8 * irregularity
    x = x + (fractal_noise_2d(h, w, period * 7, rng, octaves=2) - 0.5) * period * wander
    y = y + (fractal_noise_2d(h, w, period * 7, rng, octaves=2) - 0.5) * period * wander
    gx, gy = x / period, y / period
    i, j = np.floor(gx), np.floor(gy)
    fx, fy = gx - i, gy - j
    nx, ny = int(w / period) + 8, int(h / period) + 8
    warp_t = 1.0 + 0.7 * irregularity * (rng.random(nx) - 0.5) * 2      # thread-to-thread thickness
    weft_t = 1.0 + 0.7 * irregularity * (rng.random(ny) - 0.5) * 2
    tw = warp_t[np.clip(i.astype(int) + 4, 0, nx - 1)]
    tf = weft_t[np.clip(j.astype(int) + 4, 0, ny - 1)]
    sec_x = np.sin(np.pi * fx) ** (0.9 / tw)          # rounded thread cross-sections
    sec_y = np.sin(np.pi * fy) ** (0.9 / tf)
    over = (((i - j) % 4) < 2) if twill else (((i + j) % 2) == 0)
    cloth = np.where(over, sec_x * tw * (0.55 + 0.45 * sec_y), sec_y * tf * (0.55 + 0.45 * sec_x))
    slubs = aniso_noise(h, w, period * 6, period * 0.9, 90, rng) * 0.5 + aniso_noise(h, w, period * 6, period * 0.9, 0, rng) * 0.5
    cloth = cloth * (1.0 + 1.2 * irregularity * (slubs - 0.5))
    fibre = fractal_noise_2d(h, w, max(period * 0.6, 1.5), rng, octaves=2)
    return _normalize(0.85 * cloth + 0.15 * fibre)


def paper(h, w, scale, rng) -> np.ndarray:
    """Paper tooth: fine felted fibres."""
    fibres = sum(aniso_noise(h, w, scale * 5, scale * 0.6, a, rng, octaves=2) for a in (15, 75, 125))
    tooth = fractal_noise_2d(h, w, scale * 1.5, rng, octaves=3)
    return _normalize(0.5 * fibres / 3 + 0.5 * tooth)


def wood(h, w, scale, angle, rng) -> np.ndarray:
    """Wood grain: wavy growth rings and fine fibres along the grain."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.radians(angle)
    v = -x * np.sin(a) + y * np.cos(a)
    warp = aniso_noise(h, w, scale * 25, scale * 4, angle, rng)
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (v / scale + 2.5 * warp))
    rings = rings ** 3                                 # narrow dark latewood bands
    fibres = aniso_noise(h, w, scale * 4, 1.2, angle, rng, octaves=2)
    return _normalize(0.6 * rings + 0.4 * fibres)


def brushmarks(h, w, angle, rng) -> np.ndarray:
    """Ridges left by the brush that applied the priming: fine parallel
    striations in broad bands of varying strength."""
    fine = aniso_noise(h, w, 90, 1.8, angle, rng, octaves=2)
    bands = aniso_noise(h, w, 260, 40, angle, rng, octaves=1)
    return _normalize(fine * (0.35 + bands))
