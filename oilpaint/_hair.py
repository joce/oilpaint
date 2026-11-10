"""The hair of a brush, and the nicks of a blade, over one stroke: the
lanes its bristle clumps comb through the paint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _engine as eng
from ._geometry import _frozen

if TYPE_CHECKING:
    from .brush import BrushType

# --- the hair of a brush (see _hair_field) ----------------------------------------
# The most samples a hair pattern takes (rows times columns).
_HAIR_SAMPLES = 8.0e6
# How far the nicks of a knife's blade drift, relative to a brush's clumps.
_NICK_WANDER = 0.15
# Distance between bristle clumps (px), for the finest and the coarsest hair:
# the furrows they comb are a physical size, the same on any width of brush.
_CLUMP_PITCH = (3.0, 7.0)
# How far the clumps drift sideways along a stroke, in clump spacings (stiff
# hair; soft hair half as far).
_HAIR_WANDER = 0.65
# Lattice cells (px) of the drift, the finer drift, the regrouping and the
# bearing-down of the clumps, along the stroke, in brush widths (with a floor
# in px), and across, in clump spacings (with a floor in brush widths).
_HAIR_CELLS = (
    ((2.5, 35.0), (3.0, 0.0)),
    ((0.8, 12.0), (1.5, 0.0)),
    ((4.5, 50.0), (2.0, 0.6)),
    ((1.5, 20.0), (2.0, 0.0)),
)


def _hair_field(
    bt: BrushType,
    width: float,
    half: float,
    s0: float,
    s1: float,
    rng,
    wander: float = 1.0,
):
    """The hair of a brush over one stroke: where its bristle clumps comb the
    paint, as a field over distance along the stroke (``s0``..``s1`` px) and
    position across the bundle (px, out to a little past ``half``, its
    largest half-width), from 0 on a clump's line (a furrow) to 1 between
    clumps (a ridge); see :func:`_engine.hair_field`. Returns the field and
    where it lies: (s at its first row, x at its first column, row spacing,
    column spacing).

    The clumps are a physical size (coarser for coarse hair), not a share of
    the brush's width: a bigger brush has more of them. Along the stroke they
    drift sideways, crowd together and part (furrows merge and split), the
    brush regroups now and then (furrows end and others start), and the
    clumps bear down more or less (furrows deepen and fade): a furrow lasts
    a width or two. The field is fixed in stroke coordinates, so the steps
    of a stroke all comb the same lanes."""
    lo, hi = _CLUMP_PITCH
    pitch = hi - (hi - lo) * float(np.clip(bt.bristles, 0.0, 1.0))
    wander *= (
        _HAIR_WANDER * pitch * (0.5 + 0.5 * float(np.clip(bt.stiffness, 0.0, 1.0)))
    )
    w = max(width, 12.0)
    x_half = 1.35 * half + 1.0
    length = s1 - s0
    dx = max(0.5, 2.0 * x_half / 400.0)
    cells = np.array(
        [(max(a * w, a0), max(b * pitch, b0 * w)) for (a, a0), (b, b0) in _HAIR_CELLS]
    )
    # rows every 2 px, further apart on a long stroke, but at least four to the
    # shortest cell along it (and no more samples than _HAIR_SAMPLES)
    n_x = int(2.0 * x_half / dx) + 2
    ds = max(2.0, min(length / 1500.0, float(cells[:, 0].min()) / 4.0))
    ds = max(ds, length * n_x / _HAIR_SAMPLES)
    n_s = int(length / ds) + 2
    span_x, span_s = n_x * dx, n_s * ds
    grids = tuple(
        rng.random((int(span_s / cs) + 3, int(span_x / cx) + 3)) for cs, cx in cells
    )
    lane_cells = np.array([0.5 * pitch, max(0.2 * pitch, 1.0)])
    n_lane = int((span_x + 4.0 * wander + 2.0) / lane_cells[1]) + 3
    lanes = rng.random((4, n_lane))
    field = eng.hair_field(n_s, n_x, ds, dx, cells, grids, lanes, lane_cells, wander)
    return _frozen(field), (float(s0), -x_half, float(ds), float(dx))


# (the stippling pattern of a dab or the hair of a brush: not needed)
_NO_TEX2 = _frozen(np.zeros((1, 1)))
# (the nicks of a blade: not needed)
_NO_TEX = _frozen(np.zeros(2))
