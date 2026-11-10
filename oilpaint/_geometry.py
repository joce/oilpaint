"""The shape of a tool where it touches the canvas: brush and blade
outlines, the chord tables a footprint sweeps along a stroke, and one step's
footprint as the engine works it out."""

from __future__ import annotations

import functools
import math

import numpy as np

from . import _engine as eng


def _frozen(a: np.ndarray) -> np.ndarray:
    """``a``, made read-only: shared between strokes, or handed to the engine
    alongside arrays that are."""
    a.setflags(write=False)
    return a


_SUPERELLIPSE = {"flat": 10.0, "filbert": 3.0, "fan": 6.0, "round": 2.0, "spoon": 2.0}

# Knife blades as polygons: (u along the blade from heel -1 to tip +1, v
# across it in units of the blade's half-width, i.e. scaled by aspect).
_BLADES = {
    # a diamond trowel: widest a little behind the middle, pointed tip
    "trowel": [(-1.0, -0.3), (-0.25, -1.0), (1.0, 0.0), (-0.25, 1.0), (-1.0, 0.3)],
    # a pear / teardrop: tapering from a narrow heel to a round tip
    "pear": [(-1.0, -0.35)]
    + [
        (0.3 + 0.7 * np.cos(t), -np.sin(t))
        for t in np.linspace(np.pi / 2, -np.pi / 2, 15)
    ][::-1]
    + [(-1.0, 0.35)],
    # a long straight blade with a rounded end
    "straight": [(-1.0, -1.0)]
    + [
        (0.82 + 0.18 * np.cos(t), np.sin(t))
        for t in np.linspace(-np.pi / 2, np.pi / 2, 11)
    ]
    + [(-1.0, 1.0)],
    # a straight blade whose end is cut on a slant (a sharp toe)
    "angled": [(-1.0, -1.0), (1.0, -1.0), (0.55, 1.0), (-1.0, 1.0)],
}


def _outline(shape: str, aspect: float, n: int = 400):
    """Points (u, v) around the footprint outline, u in -1..1 along the brush's
    width (the blade's length), v across it."""
    if shape in _SUPERELLIPSE:
        e = _SUPERELLIPSE[shape]
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cu, sv = np.cos(th), np.sin(th)
        return np.sign(cu) * np.abs(cu) ** (2 / e), aspect * np.sign(sv) * np.abs(
            sv
        ) ** (2 / e)
    poly = np.asarray(_BLADES[shape], float)
    poly = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(poly * [1.0, aspect], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(t, s, poly[:, 0]), aspect * np.interp(t, s, poly[:, 1])


@functools.lru_cache(maxsize=1024)
def _footprint_area(shape: str, aspect: float) -> float:
    """Area of the footprint, in units of the half-width squared."""
    u, v = _outline(shape, aspect)
    return float(0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))


def _chord_table(shape: str, aspect: float, phi: float):
    """Geometry of a footprint swept along +t, as lookup tables over the
    lateral coordinate (all in units of the brush half-width).

    The footprint is a superellipse with half-axes 1 (along the brush's width,
    rotated by ``phi`` from the lateral direction) and ``aspect``. For each
    lateral offset l, the line through it parallel to the motion crosses the
    footprint over [c - h, c + h]. Returns (l grid, c, h, lateral half-extent, max h).

    Tables are kept for the aspect to 3 decimals and the angle to 2, and
    built for those rounded values: the same tool gets the same table
    whatever was painted before. (The arrays are shared: don't change them.)
    """
    return _chord_table_at(shape, round(aspect, 3), round(phi, 2))


@functools.lru_cache(maxsize=4096)
def _chord_table_at(shape: str, aspect: float, phi: float):
    """:func:`_chord_table` for an aspect and angle already rounded."""
    u, v = _outline(shape, aspect)
    lat = u * np.cos(phi) - v * np.sin(phi)
    al = u * np.sin(phi) + v * np.cos(phi)
    lmax = float(np.abs(lat).max())
    grid = np.linspace(-lmax, lmax, 257)
    l0, l1 = lat[None, :], np.roll(lat, -1)[None, :]
    a0, a1 = al[None, :], np.roll(al, -1)[None, :]
    den = np.where(np.abs(l1 - l0) < 1e-12, 1e-12, l1 - l0)
    t = (grid[:, None] - l0) / den
    ok = (t >= 0) & (t <= 1)
    a = a0 + t * (a1 - a0)
    lo = np.where(ok, a, np.inf).min(axis=1)
    hi = np.where(ok, a, -np.inf).max(axis=1)
    hit = np.isfinite(lo) & np.isfinite(hi)
    lo, hi = np.where(hit, lo, 0.0), np.where(hit, hi, 0.0)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    return _frozen(grid), _frozen(c), _frozen(h), lmax, float(h.max())


def _signed_floor(x, floor):
    return math.copysign(max(abs(x), floor), x if x != 0 else 1.0)


# (chord tables of a round footprint: not needed)
_NO_TABLES = (_frozen(np.zeros(2)), _frozen(np.zeros(2)), _frozen(np.zeros(2)))


def _plane(k: int, what: str) -> property:
    """Plane ``k`` of a step's footprint ``F``, as an attribute."""
    return property(lambda self: self.F[k], doc=what)


def _step_slices(st):
    """The step's part of the canvas, and where that lies in its window."""
    return (
        (slice(st.cy0, st.cy0 + st.ch), slice(st.cx0, st.cx0 + st.cw)),
        (slice(st.wy0, st.wy0 + st.ch), slice(st.wx0, st.wx0 + st.cw)),
    )


class _Step:
    """The footprint of one segment of a stroke.

    ``st`` holds its scalars (an :class:`_engine.Step`), ``F`` the per-pixel
    footprint over the window around the swept segment, which may stick out
    of the canvas (its planes are attributes, ``along`` to ``edge``; see
    :func:`_engine.footprint`). ``wsl`` crops them to the canvas part ``sl``.
    ``taps`` maps bristle columns to the pixels of that part, ``tables`` are
    the footprint's chord tables. (A knife's step not wholly on the canvas
    also has ``taps_all``, over all of its window: see _knife._knife_taps;
    and a knife's step over all of its window has no ``sl``: see
    _knife._whole_window.)"""

    sl: tuple[slice, slice] | None
    taps_all: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    along = _plane(eng.F_ALONG, "position along the motion (px)")
    lat_px = _plane(eng.F_LAT, "position across the motion (px)")
    hair = _plane(eng.F_HAIR, "the hair over the pixel, 0 (a furrow) to 1")
    aa = _plane(eng.F_AA, "how much of the pixel the footprint covers")
    profile = _plane(eng.F_PROFILE, "the tool's lateral profile")
    c = _plane(eng.F_C, "the chord's centre, along the motion (px)")
    h = _plane(eng.F_H, "the chord's half-length (px)")
    rel = _plane(eng.F_REL, "position along the chord, from its centre (px)")
    overlap = _plane(eng.F_OVERLAP, "how much of the chord the step sweeps (px)")
    expo = _plane(eng.F_EXPO, "exposure: the share of a full pass in this step")
    cover = _plane(eng.F_COVER, "coverage the step adds")
    streak = _plane(eng.F_STREAK, "how much the hair streaks lay or stir there")
    between = _plane(eng.F_BETWEEN, "the gaps between clumps or teeth")
    edge = _plane(eng.F_EDGE, "how far in from the edge of the mark (px)")

    def __init__(self, st, F, tables, taps):
        self.st, self.F, self.tables, self.taps = st, F, tables, taps
        self.sl, self.wsl = _step_slices(st)
