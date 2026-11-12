"""A tool lifted off its own paint pulls it up into peaks (stroke._dab_lift,
_knife._lift_off) by moving paint from where it falls to where it rises
(_engine.pull_up): nothing is made or lost, pigment by pigment, even where
the paint holds several colours."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes
from oilpaint import _engine as eng
from oilpaint import palette as P
from oilpaint._engine import IV, PACKED

# What is on the canvas and on the brush, channel by channel, may differ from
# what was there by this share of what the brush gave up: the worst found on
# these marks is 4.4e-8 (single-precision rounding of the pixels' paint), and
# scaling each pixel's paint, as the lift did, was 1e-4 to 0.2 off on them.
KEPT = 1e-6


def _three_colours(bt, amount):
    return (
        Brush(bt)
        .load(P.CADMIUM_RED, amount, across="left")
        .load(P.ULTRAMARINE_BLUE, amount, across="center")
        .load(P.CADMIUM_YELLOW, amount, across="right")
    )


def _on_canvas(c):
    return c.top.sum(axis=(0, 1), dtype=np.float64) + c.und.sum(
        axis=(0, 1), dtype=np.float64
    )


def _on_brush(b):
    return (b.tip + b.reservoir + b.heel).sum(axis=0)


def _assert_kept(c, brush, mark):
    canvas0, brush0 = _on_canvas(c), _on_brush(brush)
    mark()
    canvas1, brush1 = _on_canvas(c), _on_brush(brush)
    given = brush0 - brush1
    assert given[IV] > 0
    drift = (canvas1 + brush1 - canvas0 - brush0) / np.abs(given)
    worst = float(np.abs(drift).max())
    assert worst <= KEPT, f"off by {worst:.2e} of what the brush gave up"


@pytest.mark.parametrize("name", ["FLAT_HOG", "ROUND_HOG", "STIPPLER", "MOP"])
@pytest.mark.parametrize("wet", [False, True])
def test_a_multi_colour_dab_lifted_off_keeps_every_pigment(name, wet):
    """Every band of K, S, volume and body: what is on the canvas and on the
    brush after a dab of a three-colour brush lifted off into a peak is what
    there was before, to rounding (scaling each pixel of the mound made up
    to a fifth of a band, or destroyed it), on a dry canvas and on wet
    paint."""
    c = Canvas(200, 140, seed=3)
    if wet:
        c.stroke(
            [(30, 70), (170, 70)],
            P.TITANIUM_WHITE,
            amount=8,
            brush=brushes.FLAT_HOG.sized(60),
            seed=2,
        )
    brush = _three_colours(getattr(brushes, name), 25)
    _assert_kept(
        c,
        brush,
        lambda: c.stroke([(100, 70)], brush=brush, pressure=0.7, seed=2, lift=1.0),
    )


@pytest.mark.parametrize("name", ["PALETTE_KNIFE", "KNIFE_TROWEL"])
def test_a_multi_colour_knife_lifted_off_keeps_every_pigment(name):
    """The same for a knife loaded with three colours pulled up into torn
    crests as it leaves (scaling made or destroyed up to 3e-3 of a band)."""
    c = Canvas(260, 140, seed=3)
    brush = _three_colours(getattr(brushes, name), 8)
    _assert_kept(
        c,
        brush,
        lambda: c.stroke(
            [(40, 70), (200, 75)], brush=brush, pressure=0.4, seed=3, lift=1.0
        ),
    )


def test_a_lifted_multi_colour_dab_repeats_to_the_bit():
    """The same dab with the same seed paints the same bytes."""

    def dab():
        c = Canvas(200, 140, seed=3)
        brush = _three_colours(brushes.ROUND_HOG, 25)
        c.stroke([(100, 70)], brush=brush, pressure=0.7, seed=1)
        return c

    a, b = dab(), dab()
    for name in ("top", "und", "top_a", "top_id"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


# ----------------------------------------------------------------------------
# pull_up itself


def _mound(h=40, w=48, seed=0, colours=True):
    """A mound of paint (h, w, PACKED) of random thickness, its pixels each
    of their own paint (or all of one), the stroke's own paint in an
    irregular blob, and factors raising a peak off centre that keep its
    volume."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    r = np.hypot((yy - h / 2) / (0.4 * h), (xx - w / 2) / (0.4 * w))
    own = (r < 1.0) & (rng.random((h, w)) > 0.05)
    unit = rng.uniform(0.1, 3.0, (h, w, PACKED) if colours else PACKED)
    unit = np.broadcast_to(unit, (h, w, PACKED)).copy()
    unit[..., IV] = 1.0
    vol = rng.uniform(0.5, 6.0, (h, w))
    top = (unit * vol[..., None]).astype(np.float32)
    peak = np.exp(-((yy - 0.4 * h) ** 2 + (xx - 0.6 * w) ** 2) / 30.0)
    fac = _keeping_volume(top, own, np.clip(1.0 + 2.5 * (peak - 0.2), 0.05, None))
    return top, own, fac


def _keeping_volume(top, own, fac):
    V = np.where(own, top[..., IV], 0.0).astype(np.float64)
    fac = fac * V.sum() / (V * fac).sum()
    return np.where(own, fac, 1.0)


def _pulled(top, own, fac, sigma=3.0):
    out = top.copy()
    eng.pull_up(out, own, fac, sigma)
    return out


def _assert_pulled(top, own, fac, out):
    """Every own pixel ends with its factor times its volume, nothing else
    changes, and every channel's total is kept."""
    v0 = top[..., IV].astype(np.float64)
    np.testing.assert_allclose(out[..., IV][own], (v0 * fac)[own], rtol=1e-6)
    assert out[~own].tobytes() == top[~own].tobytes()
    np.testing.assert_allclose(
        out.sum(axis=(0, 1), dtype=np.float64),
        top.sum(axis=(0, 1), dtype=np.float64),
        rtol=1e-7,
    )
    assert out.min() >= 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pull_up_moves_paint_and_keeps_every_channel(seed):
    """Each pixel ends with the volume asked of it, those not the stroke's
    are left to the bit, and every channel's total is kept; the paint that
    rises is not all its own (it took in paint from around it)."""
    top, own, fac = _mound(seed=seed)
    out = _pulled(top, own, fac)
    _assert_pulled(top, own, fac, out)
    rise = own & (fac > 1.2)
    scaled = top * fac.astype(np.float32)[..., None]
    assert np.abs(out[rise] - scaled[rise]).max() > 1e-3 * np.abs(scaled).max()


def test_pull_up_scales_paint_of_one_colour_as_before():
    """Paint all of one colour is scaled pixel by pixel, to the bit as the
    lift always did it: with one composition that keeps every channel."""
    top, own, fac = _mound(colours=False)
    out = _pulled(top, own, fac)
    assert out.tobytes() == (top * fac.astype(np.float32)[..., None]).tobytes()


def test_pull_up_with_nothing_to_raise_changes_nothing():
    top, own, _ = _mound()
    assert _pulled(top, own, np.ones(own.shape)).tobytes() == top.tobytes()


def test_pull_up_between_two_pixels_moves_the_one_into_the_other():
    """A single own pixel asked to keep its volume keeps its paint; two own
    pixels, one falling and one rising: the one takes in the other's
    paint."""
    top, _, _ = _mound(h=6, w=6)
    own = np.zeros((6, 6), bool)
    own[2, 2] = True
    assert _pulled(top, own, np.ones((6, 6))).tobytes() == top.tobytes()
    own[2, 4] = True
    fac = np.ones((6, 6))
    v1, v2 = float(top[2, 2, IV]), float(top[2, 4, IV])
    fac[2, 2], fac[2, 4] = 0.5, 1.0 + 0.5 * v1 / v2
    out = _pulled(top, own, fac)
    _assert_pulled(top, own, fac, out)
    np.testing.assert_allclose(out[2, 4], top[2, 4] + 0.5 * top[2, 2], rtol=1e-6)


def test_pull_up_pools_paint_that_has_no_taker_within_reach():
    """Two islands of the stroke's paint farther apart than the plan
    reaches: all that one gives up goes to the other, through the pool, as
    the paint it was."""
    h, w = 20, 80
    top, _, _ = _mound(h=h, w=w, colours=False)
    own = np.zeros((h, w), bool)
    own[5:15, 2:12] = own[5:15, 66:76] = True
    other = _mound(h=h, w=w, seed=5, colours=False)[0]
    top[:, 40:] = other[:, 40:]  # (another colour on the right)
    fac = np.ones((h, w))
    fac[:, :40], fac[:, 40:] = 0.7, 1.3
    fac = _keeping_volume(top, own, fac)
    out = _pulled(top, own, fac, sigma=2.0)
    _assert_pulled(top, own, fac, out)
    given = (top - out)[:, :40].sum(axis=(0, 1), dtype=np.float64)
    taken = (out - top)[:, 40:].sum(axis=(0, 1), dtype=np.float64)
    # (differences of single-precision paint: rounding of the larger)
    np.testing.assert_allclose(taken, given, rtol=1e-5)
    left_unit = top[10, 5] / top[10, 5, IV]
    np.testing.assert_allclose(given / given[IV], left_unit, rtol=1e-5)


def test_pull_up_on_a_part_of_the_canvas_at_its_edge():
    """Lifted from the corner of a larger canvas (a view into it, cut off by
    the edge), it leaves the rest of the canvas as it was."""
    top, own, fac = _mound(seed=3)
    canvas = np.random.default_rng(9).uniform(0, 2, (60, 70, PACKED))
    canvas = canvas.astype(np.float32)
    canvas[:40, 22:70] = top
    before = canvas.copy()
    eng.pull_up(canvas[:40, 22:70], own, fac, 3.0)
    _assert_pulled(top, own, fac, canvas[:40, 22:70])
    canvas[:40, 22:70] = before[:40, 22:70]
    assert canvas.tobytes() == before.tobytes()
    assert _pulled(top, own, fac).tobytes() == _pulled(top, own, fac).tobytes()
