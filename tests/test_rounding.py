"""Repeatability: marks that sit on the lines rounding used to decide,
painted again with their inputs moved by an ulp (as another machine's
rounding would move them), keep the same stroke on top of every pixel and
differ in their paint by far less than a whole mark would."""

import copy

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes
from oilpaint import palette as P


def _wet_canvas():
    c = Canvas(160, 120, seed=7)
    for y in (24, 64, 104):
        c.stroke(
            [(5, y), (80, y + 5), (155, y)],
            P.YELLOW_OCHRE,
            amount=14,
            brush=brushes.FLAT_HOG.sized(40),
            pressure=0.6,
        )
    return c


_BASE = None


def _base():
    global _BASE
    if _BASE is None:
        _BASE = _wet_canvas()
    return copy.deepcopy(_BASE)


# Marks that sit on the lines rounding used to decide: whole-number
# positions and a length a whole number of steps long, a brush turned by
# exactly 90 degrees, a knife over wet paint, a dab of two paints at the
# default pressure 0.5, a stroke whose axis runs through the middles of a
# row of pixels (at half a pixel, or at 45 degrees between whole-number
# points), a curve with a segment a whole even number of pixels long.
MARKS = {
    "flat along a half-pixel axis": (
        [(20.0, 40.5), (140.0, 40.5)],
        lambda: (
            Brush(brushes.FLAT_HOG)
            .load(P.CADMIUM_RED, 4)
            .load(P.TITANIUM_WHITE, 2, across="left")
        ),
        {"pressure": 0.5},
    ),
    "flat at 45 degrees": (
        [(30.0, 20.0), (110.0, 100.0)],
        lambda: (
            Brush(brushes.FLAT_HOG)
            .load(P.CADMIUM_RED, 4)
            .load(P.TITANIUM_WHITE, 2, across="left")
        ),
        {"pressure": 0.5, "angle": 180.0},
    ),
    "knife back along a curve": (
        [(30.0, 60.0), (120.0, 60.0), (40.0, 62.0)],
        lambda: Brush(brushes.PALETTE_KNIFE).load(P.TITANIUM_WHITE, 3),
        {"pressure": 0.3},
    ),
    "comb dab at a half-pixel centre": (
        [(80.5, 60.5)],
        lambda: (
            Brush(brushes.KNIFE_COMB)
            .load(P.CADMIUM_RED, 4)
            .load(P.TITANIUM_WHITE, 2, across="left")
        ),
        {"pressure": 0.5, "angle": 180.0},
    ),
    "flat at 90 degrees": (
        [(30.0, 40.0), (110.0, 40.0)],
        lambda: Brush(brushes.FLAT_HOG.sized(32)).load(P.COBALT_BLUE, 4),
        {"pressure": 0.5, "angle": 90.0},
    ),
    "knife over wet paint": (
        [(40.0, 64.0), (120.0, 64.0)],
        lambda: Brush(brushes.PALETTE_KNIFE),
        {"pressure": 0.85, "width": 30.0},
    ),
    "dab of two paints": (
        [(80.0, 60.0)],
        lambda: (
            Brush(brushes.FILBERT_HOG)
            .load(P.CADMIUM_RED, 4)
            .load(P.TITANIUM_WHITE, 2, across="left")
        ),
        {"pressure": 0.5},
    ),
}


def _nudge(v, k):
    for _ in range(abs(k)):
        v = float(np.nextafter(v, np.inf if k > 0 else -np.inf))
    return v


def _paint(name, rng=None):
    path, brush, kw = MARKS[name]
    kw = dict(kw)
    if rng is not None:
        path = [
            (_nudge(x, rng.integers(-1, 2)), _nudge(y, rng.integers(-1, 2)))
            for x, y in path
        ]
        kw = {k: _nudge(v, rng.integers(-1, 2)) for k, v in kw.items()}
    c = _base()
    c.stroke(
        path,
        brush=brush(),
        seed=5,
        pressure=kw["pressure"],
        angle=kw.get("angle"),
        width=kw.get("width"),
    )
    return c


def _scaled_difference(a, b):
    """The largest difference over the paint, each value's as a share of the
    larger of the two (and of 1e-3, for values near 0)."""
    worst = 0.0
    for name in ("top", "und", "top_a", "und_a"):
        x, y = getattr(a, name).astype(float), getattr(b, name).astype(float)
        d = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-3)
        worst = max(worst, float(d.max()))
    return worst


@pytest.mark.parametrize("name", sorted(MARKS))
def test_a_mark_nudged_by_an_ulp_paints_the_same_to_rounding(name):
    # Measured over 12 nudges of each mark: at most 2e-7 (the flat along a
    # half-pixel axis), 1.1e-7 (the knife back along a curve), 6e-10 (the
    # knife over wet paint), the others 0; no id changed. Before, the dab
    # came out 0.73 apart (its bristle tips printed half or fully), the flat
    # at 90 degrees 1 (its sides mirrored, 173 pixels with another stroke on
    # top), the knife over wet paint 1 (21), and, as the fixes went in, the
    # flat at 45 degrees 0.40, the flat along a half-pixel axis 0.41 (its
    # middle row's paint to one ridge or the other) and the knife along a
    # curve 1 (15: the curve traced one point more or fewer).
    plain = _paint(name)
    for seed in range(3):
        nudged = _paint(name, np.random.default_rng(seed))
        assert np.array_equal(plain.top_id, nudged.top_id), seed
        assert _scaled_difference(plain, nudged) < 1e-3, seed
