"""Dabs, stippling and spatter."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes
from oilpaint import palette as P


def _dab_coverage(brush, pressure, paint="#2f5a2a", r=None):
    """Share of a dab's footprint that took paint."""
    c = Canvas(120, 120, seed=11)
    c.stroke([(60, 60)], paint, amount=6, brush=brush, pressure=pressure, seed=3)
    r = r or 0.42 * brush.width
    inner = np.hypot(*np.mgrid[-60:60, -60:60]) < r
    return float((c.thickness[inner] > 0.05).mean())


def test_a_stippler_tapped_lightly_prints_dots_that_fill_in_with_pressure():
    light, firm, hard = (_dab_coverage(brushes.STIPPLER, p) for p in (0.15, 0.35, 0.7))
    assert light < 0.5 < hard
    assert light < firm < hard
    assert hard > 0.9


def test_fewer_tips_print_bigger_separate_dots():
    def edge_share(tips):
        c = Canvas(120, 120, seed=11)
        brush = brushes.STIPPLER.with_(tips=tips)
        c.stroke([(60, 60)], "#2f5a2a", amount=6, brush=brush, pressure=0.3, seed=3)
        on = c.thickness > 0.05
        inner = on[1:-1, 1:-1]
        edge = inner & ~(on[:-2, 1:-1] & on[2:, 1:-1] & on[1:-1, :-2] & on[1:-1, 2:])
        return edge.sum() / max(inner.sum(), 1)

    # many small dots have more edge for their area than a few big ones
    assert edge_share(150) > 1.2 * edge_share(12)


def test_stipple_taps_one_brush_until_it_runs_out():
    c = Canvas(400, 80, seed=12)
    pts = [(20 + 16 * k, 40) for k in range(23)]
    brush = c.stipple(pts, P.CADMIUM_RED, amount=2, pressure=0.6)
    first = c.thickness[30:50, 8:32].sum()
    last = c.thickness[30:50, 360:384].sum()
    assert brush.volume < Brush(brushes.STIPPLER).load(P.CADMIUM_RED, 2).volume
    assert last < 0.7 * first


def test_stipple_loads_paint_for_the_width_it_taps_with():
    # loaded, before any tap
    by_width = Canvas(60, 60, seed=1).stipple([], "#336699", amount=4, width=13)
    by_size = Canvas(60, 60, seed=1).stipple(
        [], "#336699", amount=4, brush=brushes.STIPPLER.sized(13)
    )
    assert by_width.volume == pytest.approx(by_size.volume, rel=1e-6)


def test_spatter_with_no_paint_flicks_nothing():
    c = Canvas(120, 80, seed=2)
    assert c.spatter((10, 40), "#aa3322", amount=0) == 0
    assert c.spatter((10, 40), "#aa3322", amount=-1) == 0
    assert c.thickness.max() == 0


def test_paint_squeezed_from_a_soft_tool_hugs_its_edge_all_round():
    # a small mop that holds little paint, overloaded and pressed hard, squeezes
    # paint out around itself
    c = Canvas(80, 80, seed=1)
    mop = brushes.MOP.with_(capacity=0.5).sized(30)
    c.stroke([(40, 40)], P.ULTRAMARINE_BLUE, amount=6, brush=mop, pressure=0.9)
    t = c.thickness
    yy, xx = np.mgrid[0:80, 0:80]
    # no bare ring between the print and the squeezed-out paint: going out from
    # the centre, the paint thins and stops once, it doesn't stop and start again
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)):
        line = np.array([t[40 + k * dy, 40 + k * dx] for k in range(30)])
        painted = line > 0.05
        last = int(np.flatnonzero(painted).max())
        assert painted[: last + 1].all(), (dy, dx)
    # and no lips poking out at two ends: about as far all round
    reach = np.hypot(xx - 40, yy - 40)[t > 0.05]
    assert reach.max() < 1.15 * np.percentile(reach, 97)


def test_spatter_flies_the_way_it_is_flicked():
    c = Canvas(400, 300, seed=14)
    n = c.spatter((60, 150), P.CADMIUM_RED, direction=0, spread=15, reach=150, seed=1)
    ys, xs = np.nonzero(c.thickness > 0.05)
    assert n > 60 and xs.size > 0
    assert (xs < 55).mean() < 0.02  # nothing flies backwards
    assert abs(np.average(ys) - 150) < 25 and np.average(xs) > 150
    down = Canvas(300, 400, seed=14)
    down.spatter((150, 60), P.CADMIUM_RED, direction=90, spread=15, reach=150, seed=1)
    ys, xs = np.nonzero(down.thickness > 0.05)
    assert np.average(ys) > 150 and abs(np.average(xs) - 150) < 25
    more = Canvas(400, 300, seed=14).spatter((60, 150), "#000000", amount=3, seed=1)
    assert more > 2 * n


def _two_paint_dab(pressure):
    c = Canvas(120, 120, seed=11)
    c.stroke(
        [(60, 60)],
        [(P.CADMIUM_RED, 4), (P.TITANIUM_WHITE, 2, "left")],
        brush=brushes.FILBERT_HOG,
        pressure=pressure,
        seed=3,
    )
    return c


def test_a_dab_at_the_default_pressure_prints_as_one_pressed_a_hair_lighter():
    # Below 0.5 (with no glob of paint) only the longer bristle tips print.
    # That set in at once at 0.5: the gaps between the tips took half the
    # paint at 0.5 minus an ulp, all of it at 0.5.
    at, below = _two_paint_dab(0.5), _two_paint_dab(float(np.nextafter(0.5, 0.0)))
    a, b = at.top.astype(float), below.top.astype(float)
    assert np.abs(a - b).max() <= 1e-6 * np.abs(a).max()
    np.testing.assert_array_equal(at.top_id, below.top_id)


def test_tapped_more_lightly_a_dab_lays_less_paint_with_no_step():
    ps = np.linspace(0.49, 0.51, 21)
    laid = np.array([_two_paint_dab(float(p)).top[..., -2].sum() for p in ps])
    assert (np.diff(laid) >= 0).all()
    # (no step: no rise between neighbouring pressures much over the average)
    assert np.diff(laid).max() < 4 * np.diff(laid).mean()


def test_a_dab_exactly_between_two_widths_picks_the_same_pattern_either_way():
    # A dab's pattern of bristle tips is kept by its width to the nearest px:
    # 10.5 px is exactly between two, and rounded down at 10.5 and up a hair
    # above it (to even, as Python rounds). It goes up, whatever its last bit.
    def dab(width):
        c = Canvas(120, 120, seed=11)
        c.stroke(
            [(60, 60)],
            P.CADMIUM_RED,
            amount=6,
            brush=brushes.ROUND_HOG,
            width=width,
            pressure=0.35,
            seed=3,
        )
        return c.top.astype(float)

    plain = dab(10.5)
    for w in (np.nextafter(10.5, 11.0), np.nextafter(10.5, 10.0)):
        assert np.abs(dab(float(w)) - plain).max() <= 1e-4 * np.abs(plain).max()
