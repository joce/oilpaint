"""Repeatability: a last-bit difference in a stroke's inputs (as another
machine's rounding would make) changes what it paints by no more than
rounding. No pixel changes discretely: covered or not, which stroke's paint
is on top, a different number of steps or a different texture."""

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
# default pressure 0.5.
MARKS = {
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
    # Measured over 12 nudges of each mark: the knife at most 3.1e-5 (a
    # thin gap under the blade: one rounding of its level, scaled up), the
    # brush and the dab 0. Before, the dab came out 0.73 apart (its bristle
    # tips printed half or fully), the brush 1 (its sides mirrored, 173
    # pixels with another stroke on top) and the knife 1 (216).
    plain = _paint(name)
    for seed in range(4):
        nudged = _paint(name, np.random.default_rng(seed))
        assert np.array_equal(plain.top_id, nudged.top_id), seed
        assert _scaled_difference(plain, nudged) < 1e-3, seed
