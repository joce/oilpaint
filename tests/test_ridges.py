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


def _bed(width, height=H):
    """A canvas ``width`` px wide (and ``height`` high) covered with wet
    ultramarine about 5 units thick (laid lightly, then thickened)."""
    if (width, height) not in _BEDS:
        c = Canvas(width, height, seed=5)
        for k, y in enumerate(range(10, height - 5, 8)):
            c.stroke(
                [(-10, y), (width + 10, y)],
                P.ULTRAMARINE_BLUE,
                amount=30,
                brush=brushes.FLAT_HOG.sized(40),
                pressure=0.1,
                seed=k,
            )
        v = float((c.top_v + c.und_v)[30 : height - 30, 20 : width - 20].mean())
        scale = np.float32(5.0 / v)
        c.top *= scale
        c.und *= scale
        _BEDS[width, height] = c
    return _BEDS[width, height].copy()


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


def _run_by_hand(c, stroke, room=None, settle=True):
    """The stroke painted as apply_stroke paints it (with ``room`` for its
    ridges, if given; up to its settling, unless ``settle``), the run kept."""
    run = sm._Run.begin(c, stroke)
    if room is not None:
        run.room = room
        run.K = run.K._replace(room=float(room))
    run.pend = sm._pending(*run.bounds())
    for st, tables in run.steps():
        sm._brush_step(run, st, tables)
    eng.settle_pending(run.cv, run.pend, run.stroke_id)
    eng.fold_pending(run.cv, run.pend)
    if settle:
        run.settle()
    return run


def _state(c):
    return [c.top.copy(), c.und.copy(), c.top_a.copy(), c.und_a.copy(), c.top_id.copy()]


def _changed(c, before):
    return np.any(c.top + c.und != before, axis=-1)


def _settles_over(run):
    """The part of the canvas the stroke settles (see stroke._Run.settle)."""
    reach = float(run.pend.where[eng.PEND_REACH])
    return run.bounds(max(run.margin, min(reach, run.room)))


def _within(box, shape):
    y0, y1, x0, x1 = box
    inside = np.zeros(shape, bool)
    inside[y0:y1, x0:x1] = True
    return inside


def test_a_narrow_brush_has_no_more_room_than_before():
    """A brush narrow enough that its margin never reached the 7 px clip has
    as much room for its ridges as its margin, as before; a wider one has
    more, growing with the brush."""
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
        # a wide brush (its room more than its margin) whose ridges stay narrow
        (_hard(120, [(200, 5), (200, H - 5)], amount=5, pressure=0.15), True),
        # a narrow one
        (_hard(36, [(200, 5), (200, H - 5)], pressure=0.3), True),
        # a wide one whose ridges need the room
        (_hard(70, [(200, 5), (200, H - 5)]), False),
    ],
)
def test_a_stroke_whose_ridges_fit_the_old_cap_paints_as_it_always_did(stroke, fits):
    """A stroke whose ridges are no wider than the old cap (the margin less a
    px) paints the same bit for bit whatever room it has for wider ones: as
    it did before there was such room, also with a brush wide enough to have
    more. (A 70 px flat pressed hard through the bed needs the room, and
    paints differently without it.)"""
    c1, c2 = _bed(400), _bed(400)
    run = _run_by_hand(c1, stroke)
    _run_by_hand(c2, stroke, room=run.margin)
    same = all(starmap(np.array_equal, zip(_state(c1), _state(c2), strict=True)))
    assert same == fits
    assert (int(run.pend.where[eng.PEND_REACH]) > run.margin) != fits
    if stroke.brush.width > 100:
        assert run.room > run.margin + 5


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
    """A 120 px flat through the bed lays ridges past the part of the canvas
    a stroke with no more room than its margin could reach: they stay within
    the room the stroke reserved (its box of held-back paint), and the
    stroke settles out as far as they reached (it changes paint past the
    margin's reach as it settles, and nothing outside the part it settles)."""
    c = _bed(400)
    before = (c.top + c.und).copy()
    run = _run_by_hand(c, _hard(120, [(200, 5), (200, H - 5)]), settle=False)
    laid = (c.top + c.und).copy()
    changed = _changed(c, before)
    assert (changed & ~_within(run.bounds(run.margin), changed.shape)).any()
    assert not (changed & ~_within(run.bounds(), changed.shape)).any()
    run.settle()
    settled = np.any(c.top + c.und != laid, axis=-1)
    assert (settled & ~_within(run.bounds(run.margin), changed.shape)).any()
    changed = _changed(c, before)
    for box in (run.bounds(), _settles_over(run)):
        assert not (changed & ~_within(box, changed.shape)).any()


def test_where_the_room_runs_out_the_ridge_stops_there():
    """A 70 px flat heavily overloaded through the bed would push out ridges
    wider than the room a stroke reserves (as far past the mark as the
    margin's rule makes it, without its 7 px cap): before the stroke
    settles, its paint reaches out to the room and no further. A ridge w px
    wide reaches w + 0.5 px past the footprint's outline (where its mark
    fades, see _engine._side_weight), and w is at most the room less a px:
    so the paint ends within the room less half a px of the outline, in the
    last px of it."""
    thin = Paint.mix((P.TITANIUM_WHITE, 1), (P.MEDIUM, 3))
    c = _bed(280)
    before = (c.top + c.und).copy()
    stroke = _hard(70, [(140, 5), (140, H - 5)], thin, amount=400)
    run = _run_by_hand(c, stroke, settle=False)
    assert run.room > run.margin
    # the furthest the footprint's outline reaches across the stroke
    outline = max(
        st.hold.r * st.hold.lmax - 0.6 * run.K.softness * st.hold.r
        for st, _ in run.steps()
    )
    # how far across the stroke the paint it laid reaches, in the middle rows
    across = np.abs(np.arange(280) + 0.5 - 140.0)
    furthest = across[_changed(c, before)[ROWS].any(axis=0)].max()
    assert outline + run.room - 1.5 < furthest <= outline + run.room - 0.5


def _totals(c, brush):
    t = c.top.sum(axis=(0, 1), dtype=np.float64)
    t += c.und.sum(axis=(0, 1), dtype=np.float64)
    for a in (brush.tip, brush.reservoir, brush.heel):
        t += a.sum(axis=0, dtype=np.float64)
    return t


@pytest.mark.parametrize(
    ("bt", "path"),
    [
        (brushes.FLAT_HOG.sized(120), [(140, 140), (260, 260)]),  # diagonal
        (brushes.FLAT_HOG.sized(120), [(130, 120), (270, 200), (130, 280)]),  # curved
        (brushes.ROUND.sized(70), [(200, 120), (200, 280)]),  # a wide round
    ],
)
def test_wide_ridges_conserve_paint(bt, path):
    """Every channel of the paint on the canvas and the brush together is
    the same after a hard stroke through the bed as before, with ridges
    reaching past the margin (checked on a copy), all of it on the
    canvas."""
    load = 45 * bt.width / 36
    copy = _bed(400, 400)
    stroke = Stroke(path, P.TITANIUM_WHITE, amount=load, brush=bt, pressure=0.9, seed=2)
    run = _run_by_hand(copy, stroke)
    assert int(run.pend.where[eng.PEND_REACH]) > run.margin
    c = _bed(400, 400)
    brush = Brush(bt).load(P.TITANIUM_WHITE, load)
    start = _totals(c, brush)
    c.stroke(path, None, brush=brush, pressure=0.9, seed=2)
    np.testing.assert_allclose(_totals(c, brush), start, rtol=1e-6)


def test_a_wide_ridge_past_the_edge_lies_there():
    """A 120 px flat overloaded on bare canvas, run along the left edge with
    its ridge reaching past it: what it lays on the canvas, before it
    settles, is what the same stroke lays there on a canvas reaching 80 px
    further all round, to within the tolerance of the other strokes cut by
    the edge (0.5% of the paint over the canvas, 3% in all; the brush runs
    dry a little differently, as it lays its paint past the edge there).
    (Settling is left out: the edge of the canvas stops paint flowing past
    it, which on the larger canvas flows on.)"""
    off, h, w = 80, 240, 300
    x = 66.0
    out = []
    for o in (0, off):
        c = Canvas(w + 2 * o, h + 2 * o, texture=0, seed=0)
        stroke = Stroke(
            [(x + o, 20.0 + o), (x + o, h - 20.0 + o)],
            P.CADMIUM_RED,
            amount=60,
            brush=brushes.FLAT_HOG.sized(120),
            pressure=0.9,
            seed=3,
        )
        run = _run_by_hand(c, stroke, settle=False)
        if o == 0:
            assert int(run.pend.where[eng.PEND_REACH]) > run.margin
        out.append(c.wet_thickness[o : o + h, o : o + w].astype(np.float64))
    cut, big = out
    assert abs(cut.sum() / big.sum() - 1) <= 0.03
    body = np.s_[60 : h - 60]
    assert np.abs(cut[body] - big[body]).sum() <= 5e-3 * big[body].sum()


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
