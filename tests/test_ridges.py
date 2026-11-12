"""The ridges a brush pushes out at its sides are as wide as the paint in
them makes them (about as high as they are wide), up to a width that grows
with the brush: it stopped at 7 px, so that a wide brush's ridges stood up in
a narrow wall. A stroke reserves room for them and they stay within it; a
brush narrow enough that the old limit never bound paints as it always did.

Measured on a white stroke pressed hard down a bed of wet blue about 5 units
thick, both sides of the stroke and a few seeds together: one stroke's hair
pattern is chance."""

from itertools import starmap

import numpy as np
import pytest

from oilpaint import Brush, Canvas, Paint, Stroke, brushes
from oilpaint import _engine as eng
from oilpaint import palette as P
from oilpaint import stroke as sm

H = 200
ROWS = slice(50, H - 50)
SEEDS = (1, 2, 3)

_BEDS = {}


def _bed(width):
    """A canvas ``width`` px wide covered with wet ultramarine about 5 units
    thick (laid lightly, then thickened)."""
    if width not in _BEDS:
        c = Canvas(width, H, seed=5)
        for k, y in enumerate(range(10, H - 5, 8)):
            c.stroke(
                [(-10, y), (width + 10, y)],
                P.ULTRAMARINE_BLUE,
                amount=30,
                brush=brushes.FLAT_HOG.sized(40),
                pressure=0.1,
                seed=k,
            )
        v = float((c.top_v + c.und_v)[30 : H - 30, 20 : width - 20].mean())
        scale = np.float32(5.0 / v)
        c.top *= scale
        c.und *= scale
        _BEDS[width] = c
    return _BEDS[width].copy()


def _side(d, x, r, sign):
    """One side's ridge, from the mean cross-section of the paint the stroke
    added (``d``), the mark ``r`` px wide either side of ``x``: (its height
    above the bed, its width at half height). (Its crest is looked for past
    the mark: just inside it, a heavily loaded brush leaves a lip of its own
    paint.)"""
    cols = [x + sign * k for k in range(round(r) + 2, round(r) + 60)]
    top = cols[int(np.argmax(d[cols]))]
    h = float(d[top])

    def half(step):
        k = top
        while d[k + step] >= 0.5 * h:
            k += step
        return abs(k - top) + (d[k] - 0.5 * h) / (d[k] - d[k + step])

    return h, half(1) + half(-1)


def _hard(w, path, paint=P.TITANIUM_WHITE, amount=None, seed=1, pressure=0.9):
    return Stroke(
        path,
        paint,
        amount=45 * w / 36 if amount is None else amount,
        brush=brushes.FLAT_HOG.sized(w),
        pressure=pressure,
        seed=seed,
    )


def _ridges(w):
    """(height, width at half height) of the ridges either side of a white
    stroke of a ``w`` px hog flat pressed hard down the bed, over SEEDS."""
    width = max(200, 2 * w + 140)
    x = width // 2
    rows = []
    for seed in SEEDS:
        c = _bed(width)
        before = c.top_v + c.und_v
        stroke = _hard(w, [(x, 5), (x, H - 5)], seed=seed)
        # (how far the mark reaches either side, the brush splayed)
        run = sm._Run.begin(c.copy(), stroke)
        r = float(run.radii.max()) * run.reach
        c.paint(stroke)
        d = (c.top_v + c.und_v - before)[ROWS].mean(axis=0).astype(np.float64)
        rows += [_side(d, x, r, sign) for sign in (-1, 1)]
    return np.array(rows).mean(axis=0)


def test_a_wide_brush_s_ridge_is_wider_and_lower_than_the_cap_made_it():
    """A 120 px hog flat pressed hard through the bed pushes out ridges
    wider than the 7 px cap let them be, and so lower and less of a wall:
    nearly twice as wide as a 36 px flat's (the cap held them to under one
    and a half times), and at least 0.4 times as wide as they are high (1
    unit of paint ~ 0.6 px; 0.36 under the cap)."""
    h36, half36 = _ridges(36)
    h120, half120 = _ridges(120)
    assert half120 > 1.7 * half36
    assert half120 >= 0.42 * (0.6 * h120)
    assert half36 >= 0.6 * (0.6 * h36)


def _run_by_hand(c, stroke, room=None):
    """The stroke painted as apply_stroke paints it (with ``room`` for its
    ridges, if given), the run kept."""
    run = sm._Run.begin(c, stroke)
    if room is not None:
        run.room = room
        run.K = run.K._replace(room=float(room))
    run.pend = sm._pending(*run.bounds())
    for st, tables in run.steps():
        sm._brush_step(run, st, tables)
    eng.settle_pending(run.cv, run.pend, run.stroke_id)
    eng.fold_pending(run.cv, run.pend)
    run.settle()
    return run


def _state(c):
    return [c.top.copy(), c.und.copy(), c.top_a.copy(), c.und_a.copy(), c.top_id.copy()]


def test_a_narrow_brush_has_no_more_room_than_before():
    """Up to about 55 px for a hog flat (40 px for a round) the room a
    stroke has for its ridges is its margin, as before; wider, it grows with
    the brush."""
    for bt, room_is_margin in (
        (brushes.FLAT_HOG.sized(16), True),
        (brushes.FLAT_HOG.sized(50), True),
        (brushes.ROUND.sized(36), True),
        (brushes.FLAT_HOG.sized(70), False),
        (brushes.FLAT_HOG.sized(120), False),
    ):
        stroke = Stroke([(30, 100), (170, 100)], P.TITANIUM_WHITE, amount=10, brush=bt)
        run = sm._Run.begin(Canvas(200, H, seed=1), stroke)
        assert (run.room == run.margin) == room_is_margin
    knife = Stroke([(30, 100), (170, 100)], brush=brushes.SCRAPER)
    run = sm._Run.begin(Canvas(200, H, seed=1), knife)
    assert run.room == run.margin


@pytest.mark.parametrize(
    ("stroke", "fits"),
    [
        (_hard(36, [(100, 5), (100, H - 5)], pressure=0.3), True),
        (
            Stroke(
                [(100, 5), (60, 120), (110, H - 5)],
                P.TITANIUM_WHITE,
                amount=20,
                brush=brushes.ROUND.sized(24),
                pressure=0.3,
                seed=2,
            ),
            True,
        ),
        (_hard(70, [(100, 5), (100, H - 5)]), False),
    ],
)
def test_a_stroke_whose_ridges_fit_the_old_cap_paints_as_it_always_did(stroke, fits):
    """A stroke whose ridges are no wider than the old cap (the margin less a
    px) paints the same bit for bit whatever room it has for wider ones: as
    it did before there was such room. (A 70 px flat pressed hard through the
    bed needs the room, and paints differently without it.)"""
    c1, c2 = _bed(200), _bed(200)
    run = _run_by_hand(c1, stroke)
    _run_by_hand(c2, stroke, room=run.margin)
    same = all(starmap(np.array_equal, zip(_state(c1), _state(c2), strict=True)))
    assert same == fits
    assert (int(run.pend.where[eng.PEND_REACH]) > run.margin) != fits


def test_the_stroke_records_what_it_meets_before_its_ridges_land():
    """Where a stroke's ridges land past its usual margin, it has recorded
    how far the paint there can be ploughed before they land (see
    _engine._meet): every pixel the stroke changed holds what the bed held
    there, as a stroke turning back over its own ridges must find it."""
    c = _bed(400)
    v = (c.top_v + c.und_v).astype(np.float64)
    body = (c.top[..., eng.IB] + c.und[..., eng.IB]) / np.maximum(v, 1e-30)

    def ss(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)

    want = ss(eng.PLOUGH_T0, eng.PLOUGH_T1, v) * ss(eng.PLOUGH_B0, eng.PLOUGH_B1, body)
    before = (c.top + c.und).copy()
    run = sm._Run.begin(c, _hard(120, [(80, 40), (320, 100), (80, 160)]))
    run.pend = sm._pending(*run.bounds())
    for st, tables in run.steps():
        sm._brush_step(run, st, tables)
    eng.settle_pending(run.cv, run.pend, run.stroke_id)
    where, met = run.pend.where, run.pend.met
    y0, x0 = where[eng.PEND_Y0], where[eng.PEND_X0]
    box = np.s_[y0 : y0 + met.shape[0], x0 : x0 + met.shape[1]]
    changed = np.any(c.top + c.und != before, axis=-1)[box]
    assert where[eng.PEND_REACH] > run.margin + 4
    assert (met[changed] >= 0).all()
    np.testing.assert_allclose(met[met >= 0], want[box][met >= 0], atol=1e-6)


def test_a_wide_ridge_stays_within_the_room_the_stroke_reserves():
    """A 70 px flat through the bed pushes out ridges reaching well past its
    usual margin: they stay within the room the stroke reserved (nothing
    changes outside its box of held-back paint), and the stroke settles all
    the way out to them (nothing outside the part it settles changes)."""
    c = _bed(280)
    before = (c.top + c.und).copy()
    run = _run_by_hand(c, _hard(70, [(140, 5), (140, H - 5)]))
    reach = int(run.pend.where[eng.PEND_REACH])
    assert run.margin < reach <= np.ceil(run.room)
    changed = np.any(c.top + c.und != before, axis=-1)
    for box in (run.bounds(), run.bounds(max(run.margin, reach))):
        y0, y1, x0, x1 = box
        inside = np.zeros_like(changed)
        inside[y0:y1, x0:x1] = True
        assert not (changed & ~inside).any()
    # (and it did need the room: the ridges reach past the usual bounds)
    y0, y1, x0, x1 = run.bounds(run.margin)
    assert changed[:, : x0 - 1].any() or changed[:, x1 + 1 :].any()


def test_where_the_room_runs_out_the_ridge_stops_there():
    """A 70 px flat heavily overloaded through the bed would push out ridges
    wider than the room a stroke reserves (as far past the mark as the
    margin's rule makes it, without its 7 px cap): they reach that far, and
    no further."""
    thin = Paint.mix((P.TITANIUM_WHITE, 1), (P.MEDIUM, 3))
    run = _run_by_hand(_bed(280), _hard(70, [(140, 5), (140, H - 5)], thin, amount=400))
    assert run.room > run.margin
    reach = int(run.pend.where[eng.PEND_REACH])
    assert reach == int(np.ceil(run.room))


def _totals(c, brush):
    t = c.top.sum(axis=(0, 1), dtype=np.float64)
    t += c.und.sum(axis=(0, 1), dtype=np.float64)
    for a in (brush.tip, brush.reservoir, brush.heel):
        t += a.sum(axis=0, dtype=np.float64)
    return t


@pytest.mark.parametrize(("name", "w"), [("FLAT_HOG", 70), ("ROUND_HOG", 24)])
def test_wide_ridges_conserve_paint(name, w):
    """Every channel of the paint on the canvas and the brush together is
    the same after a hard stroke through the bed as before, however wide its
    ridges."""
    width = max(200, 2 * w + 140)
    c = _bed(width)
    brush = Brush(getattr(brushes, name).sized(w)).load(P.TITANIUM_WHITE, 45 * w / 36)
    start = _totals(c, brush)
    x = width // 2
    c.stroke([(x, 60), (x, H - 40)], None, brush=brush, pressure=0.9, seed=2)
    np.testing.assert_allclose(_totals(c, brush), start, rtol=1e-6)


@pytest.mark.parametrize(
    "path",
    [
        [(0.0, 5), (0.0, H - 5)],  # half off the canvas
        [(37.0, 5), (37.0, H - 5)],  # its ridge past the edge
        [(20, 15), (20, H - 60)],  # set down in a corner
    ],
)
def test_a_ridge_past_the_edge_goes_off_the_canvas(path):
    """A wide flat run along the edge of the canvas, half off it, just
    inside it with its ridge reaching past the edge, or set down in a
    corner: the paint on the canvas and the brush is at most what it was,
    channel by channel (the rest went off the canvas), and none is
    negative."""
    c = _bed(280)
    brush = Brush(brushes.FLAT_HOG.sized(70)).load(P.TITANIUM_WHITE, 45 * 70 / 36)
    start = _totals(c, brush)
    c.stroke(path, None, brush=brush, pressure=0.9, seed=3)
    after = _totals(c, brush)
    assert np.all(after <= start * (1 + 1e-9) + 1e-9)
    assert after[eng.IV] < start[eng.IV]
    assert c.top.min() > -1e-5 and c.und.min() > -1e-5
    assert np.isfinite(c.top).all() and np.isfinite(c.und).all()
