"""A tool lifted off its own paint pulls it up into peaks (stroke._dab_lift,
_knife._lift_off) by moving paint from where it falls to where it rises
(_engine.pull_up): nothing is made or lost, pigment by pigment, even where
the paint holds several colours, and the paint a peak takes in is the paint
around it."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes
from oilpaint import _engine as eng
from oilpaint import palette as P
from oilpaint._engine import IV, PACKED, N

# What is on the canvas and on the brush after a whole mark, channel by
# channel, may differ from what was there by this share of what the brush
# gave up: the stroke's every step rounds its pixels to single precision
# (the worst found on these marks is 4.4e-8), and scaling each pixel's paint,
# as the lift did, was 6e-4 to 0.2 off on them.
KEPT = 1e-6

# What pull_up itself may be off by, relative to a channel's total or to a
# pixel's volume: it works in double precision and writes each pixel once,
# a pixel that falls as fl(p * fl(f)), one that rises as fl(p + fl(t)) with
# p, t >= 0. Each of those is within 2u of the exact value (u = 2^-24, the
# rounding of single precision; for p + t, u t + u (p + t) <= 2u (p + t)),
# so the total of a channel, all of whose terms are >= 0, is too; 1e-12
# covers the double-precision sums.
ROUND = 2 * 2.0**-24 + 1e-12


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


def _dab(c, bt):
    brush = _three_colours(bt, 25)
    return brush, lambda: c.stroke(
        [(100, 70)], brush=brush, pressure=0.7, seed=2, lift=1.0
    )


def _knife(c, bt):
    brush = _three_colours(bt, 8)
    return brush, lambda: c.stroke(
        [(40, 70), (200, 75)], brush=brush, pressure=0.4, seed=3, lift=1.0
    )


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
    _assert_kept(c, *_dab(c, getattr(brushes, name)))


@pytest.mark.parametrize("name", ["PALETTE_KNIFE", "KNIFE_TROWEL"])
def test_a_multi_colour_knife_lifted_off_keeps_every_pigment(name):
    """The same for a knife loaded with three colours pulled up into torn
    crests as it leaves (scaling made or destroyed 6e-4 and 1.5e-3 of a
    band)."""
    c = Canvas(260, 140, seed=3)
    _assert_kept(c, *_knife(c, getattr(brushes, name)))


def _spread(paint, whole):
    """The mean distance (by volume) of what is in a unit of each of
    ``paint`` (m, PACKED) from ``whole``, relative to it."""
    unit = paint[:, :N] / paint[:, IV : IV + 1]
    dist = np.abs(unit - whole).sum(axis=1) / np.abs(whole).sum()
    return float((dist * paint[:, IV]).sum() / paint[:, IV].sum())


def _locality(calls):
    """How local the paint taken in by the pixels that rose is, over the
    lifts in ``calls`` (the paint before and after each, and the factors):
    the mean distance of what each took in (per unit of it) from what all
    those that fell gave up together, over the mean distance of what each
    of those gave up from that. Near 0: each took in the same mixture of the
    whole mark; near 1 or more: each took in paint as varied as the mark's
    own, its neighbours'."""
    num = den = 0.0
    for before, after, fac in calls:
        b64, a64 = before.astype(np.float64), after.astype(np.float64)
        rise = (fac > 1.0) & (b64[..., IV] > 0)
        fall = (fac < 1.0) & (b64[..., IV] > 0)
        took = a64[rise] - b64[rise]
        gave = b64[fall] - a64[fall]
        whole = gave.sum(axis=0)
        whole = whole[:N] / whole[IV]
        num += _spread(took, whole)
        den += _spread(gave, whole)
    return num / den


@pytest.mark.parametrize("tool", ["dab", "knife"])
def test_a_peak_takes_in_the_paint_around_it(monkeypatch, tool):
    """The paint a dab's peak or a knife's crests take in comes from near
    them, not as a mixture of the whole mark: the scale the lift moves paint
    over (a peak's size, a crest's spacing) holds. (Here 0.57 for the dab
    and 0.72 for the knife; off by three times either way 0.26 and 0.39 at
    most, by ten times 0.04.)"""
    calls = []
    pull_up = eng.pull_up

    def watched(top, fac, sigma):
        before = top.copy()
        pull_up(top, fac, sigma)
        calls.append((before, top.copy(), fac.copy()))

    monkeypatch.setattr(eng, "pull_up", watched)
    if tool == "dab":
        c = Canvas(200, 140, seed=3)
        _dab(c, brushes.FLAT_HOG.sized(40))[1]()
    else:
        c = Canvas(260, 140, seed=3)
        _knife(c, brushes.KNIFE_TROWEL)[1]()
    assert len(calls) == 1
    assert _locality(calls) > 0.45


# ----------------------------------------------------------------------------
# pull_up itself


def _mound(rng, h, w, colours=True, zero=0.0):
    """Paint (h, w, PACKED) of random thickness, each pixel of its own paint
    (or all of one), a share ``zero`` of the pixels holding no paint but a
    trace of pigment; the stroke's own pixels at random, and random factors
    for them that keep their volume (1 elsewhere)."""
    own = rng.random((h, w)) < rng.uniform(0.2, 1.0)
    unit = rng.uniform(0.01, 3.0, (h, w, PACKED) if colours else PACKED)
    unit = np.broadcast_to(unit, (h, w, PACKED)).copy()
    unit[..., IV] = 1.0
    vol = rng.uniform(0.01, 6.0, (h, w))
    top = (unit * vol[..., None]).astype(np.float32)
    empty = rng.random((h, w)) < zero
    top[empty, IV] = 0.0  # (pigment with no volume: rounding leaves such)
    fac = np.clip(1.0 + rng.normal(0.0, rng.uniform(0.05, 1.5), (h, w)), 0.05, None)
    return top, _keeping_volume(top, own, fac)


def _keeping_volume(top, own, fac):
    V = np.where(own, top[..., IV], 0.0).astype(np.float64)
    if V.sum() > 0:
        fac = fac * V.sum() / max(float((V * fac).sum()), 1e-300)
    return np.where(own, fac, 1.0)


def _pulled(top, fac, sigma=3.0):
    out = top.copy()
    eng.pull_up(out, fac, sigma)
    return out


def _assert_pulled(top, fac, out):
    """Every pixel with paint ends with its factor times its volume, nothing
    else changes (a pixel left alone, or with no paint, to the bit), every
    channel's total is kept, nothing is less than nothing and all is finite."""
    assert np.isfinite(out).all()
    assert out.min() >= 0
    v0 = top[..., IV].astype(np.float64)
    moved = (fac != 1.0) & (v0 > 0)
    np.testing.assert_allclose(
        out[..., IV][moved], (v0 * fac)[moved], rtol=ROUND, atol=0
    )
    assert out[~moved].tobytes() == top[~moved].tobytes()
    np.testing.assert_allclose(
        out.sum(axis=(0, 1), dtype=np.float64),
        top.sum(axis=(0, 1), dtype=np.float64),
        rtol=ROUND,
        atol=0,
    )


@pytest.mark.parametrize("seed", range(6))
def test_pull_up_keeps_every_channel_on_any_mound(seed):
    """Small mounds of every shape (rows and columns one pixel across too),
    any share of the stroke's own paint, pixels with no paint, a reach from
    a hair to far more than the mound, lifted from a part of a larger
    canvas with other paint all round: every channel's total is kept to
    rounding, each pixel with paint ends with its factor times its volume,
    and nothing else, in the mound or round it, changes."""
    rng = np.random.default_rng(seed)
    for _ in range(50):
        h, w = (int(n) for n in rng.integers(1, 30, 2))
        if rng.random() < 0.2:
            h, w = (1, w) if rng.random() < 0.5 else (h, 1)
        top, fac = _mound(rng, h, w, rng.random() < 0.8, rng.uniform(0, 0.4))
        sigma = float(10 ** rng.uniform(-2, 1.7))
        oy, ox = (int(n) for n in rng.integers(0, 4, 2))
        canvas = rng.uniform(0, 2, (h + 5, w + 6, PACKED)).astype(np.float32)
        canvas[oy : oy + h, ox : ox + w] = top
        before = canvas.copy()
        part = canvas[oy : oy + h, ox : ox + w]
        eng.pull_up(part, fac, sigma)
        _assert_pulled(top, fac, part.copy())
        part[:] = top
        assert canvas.tobytes() == before.tobytes()


def test_pull_up_gives_the_same_bytes_wherever_the_mound_is():
    """The same mound lifted from different places in a larger canvas, with
    other paint round it each time, comes out the same to the bit."""
    rng = np.random.default_rng(4)
    top, fac = _mound(rng, 24, 31)
    outs = []
    for oy, ox, seed in ((0, 0, 1), (7, 3, 2), (2, 11, 3)):
        canvas = np.random.default_rng(seed).uniform(0, 2, (40, 50, PACKED))
        canvas = canvas.astype(np.float32)
        canvas[oy : oy + 24, ox : ox + 31] = top
        big = np.ones((40, 50))
        big[oy : oy + 24, ox : ox + 31] = fac
        eng.pull_up(canvas, big, 3.0)
        outs.append(canvas[oy : oy + 24, ox : ox + 31].tobytes())
    assert outs[0] == outs[1] == outs[2]


@pytest.mark.parametrize("trace", [1e-8, 1e-3])
def test_pull_up_scales_paint_of_one_colour_and_moves_any_other(trace):
    """Paint all of one colour (to rounding: a trace of 1e-8 apart) is
    scaled pixel by pixel, to the bit as the lift always did it: with one
    composition that keeps every channel. Paint a thousandth apart is
    moved, and keeps every channel too."""
    rng = np.random.default_rng(0)
    top, fac = _mound(rng, 30, 36, colours=False)
    jitter = 1.0 + trace * rng.uniform(-1, 1, top.shape)
    jitter[..., IV] = 1.0
    top = (top * jitter).astype(np.float32)
    out = _pulled(top, fac)
    _assert_pulled(top, fac, out)
    scaled = (top * fac.astype(np.float32)[..., None]).tobytes()
    assert (out.tobytes() == scaled) == (trace < 1e-6)


def test_pull_up_between_two_pixels_moves_the_one_into_the_other():
    """Two pixels of paint, one falling and one rising: the one takes in the
    other's paint; a pixel with no volume but a trace of pigment, asked to
    fall, keeps it."""
    top = np.zeros((5, 6, PACKED), np.float32)
    top[2, 1] = np.linspace(1, 3, PACKED)
    top[2, 4] = np.linspace(3, 1, PACKED)
    top[2, 2, :N] = 0.5  # (no volume)
    fac = np.ones((5, 6))
    v1, v2 = float(top[2, 1, IV]), float(top[2, 4, IV])
    fac[2, 1], fac[2, 2], fac[2, 4] = 0.5, 0.3, 1.0 + 0.5 * v1 / v2
    out = _pulled(top, fac)
    _assert_pulled(top, fac, out)
    np.testing.assert_allclose(out[2, 4], top[2, 4] + 0.5 * top[2, 1], rtol=ROUND)
    assert out[2, 2].tobytes() == top[2, 2].tobytes()


def test_pull_up_with_none_to_rise_or_no_reach_changes_nothing_or_pools():
    """With no pixel to rise (or none to fall) nothing changes; with no
    reach at all, or none that makes sense, all that falls goes through the
    pool and every channel is still kept."""
    rng = np.random.default_rng(2)
    top, fac = _mound(rng, 12, 14)
    falling = np.where(fac > 1.0, 1.0, fac)
    assert _pulled(top, falling).tobytes() == top.tobytes()
    rising = np.where(fac < 1.0, 1.0, fac)
    assert _pulled(top, rising).tobytes() == top.tobytes()
    for sigma in (0.0, -1.0, float("nan"), float("inf"), 1e-300):
        _assert_pulled(top, fac, _pulled(top, fac, sigma))


def _halves(h=30, w=120, gap=60):
    """Paint of one colour on the left of ``gap``, another on the right."""
    top = np.empty((h, w, PACKED), np.float32)
    top[:, :gap] = np.linspace(1, 3, PACKED)
    top[:, gap:] = np.linspace(3, 1, PACKED)
    top[..., IV] = 2.0
    return top


def _peak(fac, cy, cx, size=3.0, ring=9.0):
    """A peak at (cy, cx) pulled up from a ring round it, and nothing
    beyond: the factors keep the volume of a uniform layer."""
    yy, xx = np.mgrid[0 : fac.shape[0], 0 : fac.shape[1]]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    up = np.exp(-r2 / (2 * size**2))
    near = r2 <= ring**2
    up = np.where(near, up - up[near].mean(), 0.0)
    return fac + 0.8 * up / up.max()


def _share_of_right(top, out, rise):
    took = (out.astype(np.float64) - top)[rise].sum(axis=0)
    left, right = top[0, 0].astype(np.float64), top[0, -1].astype(np.float64)
    # took = l * left + r * right, by least squares over the channels
    coef = np.linalg.lstsq(np.stack([left, right], 1), took, rcond=None)[0]
    return coef[1] / coef.sum()


def test_pull_up_moves_paint_only_between_near_pixels():
    """Two peaks, each pulled up from the paint round it, one in each half
    of a mound of two colours and far apart: each takes in its own half's
    colour, almost all of it. A peak by the border between them takes in
    some of each."""
    top = _halves()
    fac = _peak(_peak(np.ones(top.shape[:2]), 15, 25), 15, 95)
    out = _pulled(top, fac, sigma=3.0)
    _assert_pulled(top, fac, out)
    rise = fac > 1.0
    left, right = rise.copy(), rise.copy()
    left[:, 60:], right[:, :60] = False, False
    assert _share_of_right(top, out, left) < 0.02
    assert _share_of_right(top, out, right) > 0.98
    fac = _peak(np.ones(top.shape[:2]), 15, 58)
    out = _pulled(top, fac, sigma=3.0)
    assert 0.1 < _share_of_right(top, out, fac > 1.0) < 0.9


def test_pull_up_pools_paint_that_has_no_taker_within_reach():
    """Two islands of the stroke's paint farther apart than the plan
    reaches: all that one gives up goes to the other, through the pool, as
    the paint it was."""
    top = _halves(20, 80, 40)
    fac = np.ones((20, 80))
    fac[5:15, 2:12], fac[5:15, 66:76] = 0.7, 1.3
    out = _pulled(top, fac, sigma=2.0)
    _assert_pulled(top, fac, out)
    given = (top.astype(np.float64) - out)[:, :40].sum(axis=(0, 1))
    taken = (out.astype(np.float64) - top)[:, 40:].sum(axis=(0, 1))
    # (differences of single-precision paint: rounding of the larger)
    np.testing.assert_allclose(taken, given, rtol=1e-5)
    np.testing.assert_allclose(given / given[IV], top[0, 0] / top[0, 0, IV], rtol=1e-5)
