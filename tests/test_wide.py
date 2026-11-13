"""A wide brush pressed hard through thick paint can't squeeze the paint
under its middle out to its sides: under a plate w wide the paint has w/2
to travel, and the rate a pressure thins it falls as 1/w^2, so only the
paint near an edge gets out (see _engine.EDGE_FREE). What it can't push
aside stays under it: the paint it carries stays on it and goes on thicker,
the wet paint it ploughs stays where it was.

Before, a 120 px flat threw all of it out to its sides: down the bed here
its ridges held 423 units x px a side against 55 for a 36 px flat (they
grew with the square of the load across the brush), stood 35 units high
(21 px) and 6.3 px wide at half height, and the brush set down left a wall.

Measured on white strokes 150 px long, pressed hard (0.9) and loaded
45 w / 36, down a flat bed of wet ultramarine 5 units thick (one seed) or
on bare canvas (three seeds: its ridges vary much more from one stroke to
the next): the mean cross-section over their middle rows, both sides."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, Stroke, brushes
from oilpaint import _engine as eng
from oilpaint import palette as P
from oilpaint import stroke as sm

H = 330
Y0, Y1 = 100, 250  # the strokes run down from Y0 to Y1
ROWS = slice(145, 215)  # their middle rows
# Each measure is taken over five strokes (seeds): one stroke's hair pattern
# is chance (a 120 px flat's ridges stand 14 to 31 units high down the bed
# from one to the next). The bounds sit on the spread of the means over ten.
SEEDS = (1, 2, 3, 4, 5)
IV = eng.IV


def _canvas(width, bed, height=H):
    """A canvas ``width`` px wide (``height`` high), bare or covered with wet
    ultramarine 5 units thick (laid all at once: a flat bed)."""
    c = Canvas(width, height, seed=5)
    if bed:
        b = Brush(brushes.FLAT_HOG).load(P.ULTRAMARINE_BLUE, 20)
        pk = b.reservoir.sum(axis=0) + b.tip.sum(axis=0)
        c.und[:] = (pk * (5.0 / pk[IV])).astype(np.float32)
        c.und_a[:] = 1.0
    return c


def _totals(c, brush):
    t = c.top.sum(axis=(0, 1), dtype=np.float64)
    t += c.und.sum(axis=(0, 1), dtype=np.float64)
    for a in (brush.tip, brush.reservoir, brush.heel):
        t += a.sum(axis=0, dtype=np.float64)
    return t


_STROKES = {}


def _stroke(w, bed, bt=brushes.FLAT_HOG, seed=1):
    """A hard white stroke of ``bt`` sized ``w`` down the middle of a
    canvas (with a bed or bare): (canvas, wet volume before, its centre
    column, the totals of canvas and brush before and after)."""
    key = (w, bed, bt.name, seed)
    if key not in _STROKES:
        width = 2 * w + 160
        x = width // 2
        c = _canvas(width, bed)
        before = (c.top_v + c.und_v).astype(np.float64)
        brush = Brush(bt.sized(w)).load(P.TITANIUM_WHITE, 45 * w / 36)
        start = _totals(c, brush)
        c.stroke([(x, Y0), (x, Y1)], None, brush=brush, pressure=0.9, seed=seed)
        _STROKES[key] = (c, before, x, start, _totals(c, brush))
    return _STROKES[key]


def _ridge(d, x, w, sign):
    """One side's ridge in the mean cross-section ``d`` of the paint a stroke
    of a ``w`` px brush centred on column ``x`` added: (height, width at
    half height, steepest fall outwards per px, area: the paint past the
    mark, units x px). Its crest is looked for from just inside the edge of
    the mark out."""
    edge = x + sign * w / 2
    cols = np.array([int(edge + sign * k) for k in range(-3, 60)])
    top = cols[int(np.argmax(d[cols]))]
    h = float(d[top])

    def half(step):
        k = top
        while d[k + step] >= 0.5 * h:
            k += step
        return abs(k - top) + (d[k] - 0.5 * h) / (d[k] - d[k + step])

    out = d[top::sign][:40]
    slope = float(np.max(-np.diff(out)))
    area = float(np.clip(d[cols[3:]], 0, None).sum())
    return h, half(1) + half(-1), slope, area


def ridges(w, bed, bt=brushes.FLAT_HOG):
    """The two side ridges of a hard stroke of a ``w`` px ``bt`` (see
    _stroke), averaged over both sides and SEEDS: (height, ratio of width
    at half height to height in px (1 unit ~ 0.6 px), steepest outward
    slope, area a side)."""
    rows = []
    for seed in SEEDS:
        c, before, x, _, _ = _stroke(w, bed, bt, seed)
        d = ((c.top_v + c.und_v).astype(np.float64) - before)[ROWS].mean(axis=0)
        rows += [_ridge(d, x, w, s) for s in (-1, 1)]
    h, half, slope, area = np.mean(rows, axis=0)
    return h, half / (0.6 * h), slope, area


def interior(w, bed, bt=brushes.FLAT_HOG):
    """The paint a hard stroke of a ``w`` px brush leaves over the middle
    half of its mark (units, on top of what was there), and how much of
    the bed is left under it (the mean of the lower layer), over SEEDS."""
    rows = []
    for seed in SEEDS:
        c, before, x, _, _ = _stroke(w, bed, bt, seed)
        mid = slice(int(x - w / 4), int(x + w / 4) + 1)
        v = (c.top_v + c.und_v).astype(np.float64)[ROWS, mid] - before[ROWS, mid]
        rows.append((v.mean(), c.und_v[ROWS, mid].mean()))
    added, below = np.mean(rows, axis=0)
    return float(added), float(below)


def bead(w, bed, bt=brushes.FLAT_HOG):
    """The height of the bead where a hard stroke of a ``w`` px brush was set
    down: the highest of the mean, over the middle half of the mark, of the
    paint it added in the rows behind and about where it started; the mean
    over SEEDS."""
    tops = []
    for seed in SEEDS:
        c, before, x, _, _ = _stroke(w, bed, bt, seed)
        mid = slice(int(x - w / 4), int(x + w / 4) + 1)
        d = ((c.top_v + c.und_v).astype(np.float64) - before)[:, mid].mean(axis=1)
        tops.append(d[Y0 - 30 : Y0 + 8].max())
    return float(np.mean(tops))


@pytest.mark.parametrize("bed", [True, False])
@pytest.mark.parametrize("w", [70, 120])
def test_a_wide_brush_s_ridges_keep_a_small_brush_s_proportions(w, bed):
    """A 70 or 120 px flat pressed hard, down the bed or on bare canvas, pushes
    out ridges whose width at half height to their height (1 unit ~ 0.6 px)
    is more than 0.6 times a 36 px flat's down the bed (1.19; 1.05 over
    seeds 6 to 10), no higher than 2.9 times its (9.0 units) and whose
    steepest outward slope (each side's steepest fall in the mean
    cross-section) is under 3.2 units a px. Means over five seeds, seeds 1
    to 5 (and 6 to 10): ratio 0.91 (0.73), 0.79 (0.81) down the bed at 70
    and 120 px, 2.8 (1.5), 1.21 (1.34) on bare canvas; height 14 (18), 24
    (23), 8.6 (11), 17 (18); slope 2.3 (3.0), 2.8 (2.9), 1.5 (1.9), 2.1
    (2.3). Before: ratio 0.65 (0.53) and 0.35 (0.32) down the bed, 0.49
    (0.43) at 120 px on bare canvas; height 53 at 120 px down the bed;
    slope 3.2 (4.2) and 6.3 (6.8), 4.7 (5.3) at 120 px on bare canvas. (On
    bare canvas a 70 px flat's ridges were within these before too.)"""
    h, ratio, slope, _ = ridges(w, bed)
    h36, ratio36, _, _ = ridges(36, True)
    assert ratio > 0.6 * ratio36
    assert slope < 3.2
    assert h < 2.9 * h36


def test_a_wide_round_s_ridges_keep_their_proportions_too():
    """The same for a 120 px hog round down the bed, against a 36 px round
    (ratio 1.02; 0.98 over seeds 6 to 10): ratio 1.09 (1.14), against 0.55
    (0.50) before; slope 2.2 (2.2), against 4.5 (4.5)."""
    _, ratio, slope, _ = ridges(120, True, brushes.ROUND_HOG)
    _, ratio36, _, _ = ridges(36, True, brushes.ROUND_HOG)
    assert ratio > 0.75 * ratio36
    assert slope < 3.2


@pytest.mark.parametrize("bed", [True, False])
def test_a_wide_brush_s_ridges_grow_more_slowly(bed):
    """The paint in the ridges of a hard stroke, loaded in proportion to its
    width (45 w / 36), grows more slowly with the brush than it did: from 70
    to 120 px less than 3.6 times (seeds 1 to 5, and 6 to 10: 2.4 and 1.9
    down the bed, 3.0 and 2.3 on bare canvas; it was 3.8 and 3.1, 5.0 and
    3.8, from the paint under its middle), and down the bed from 36 px less
    than 7 times (5.3 and 5.1; 12.2 and 12.0 before). It still grows (more
    than 1.2 times from 70 px). (A 36 px flat's ridges on bare canvas run
    into its own paint: no measure.)"""
    a70 = ridges(70, bed)[3]
    a120 = ridges(120, bed)[3]
    assert a120 < 3.6 * a70
    assert a120 > 1.2 * a70
    if bed:
        assert a120 < 7.0 * ridges(36, bed)[3]


def test_the_paint_held_back_stays_under_a_wide_brush():
    """What a 120 px flat can't push out to its sides down the bed stays under
    it: its stroke is thicker in the middle than when it all went aside
    (5.4 units more than the bed's 5, over seeds 1 to 5; 5.0 over seeds 6
    to 10; 3.4 to 6.3 stroke by stroke; against 3.2 and 2.8 before), and
    holds more of the bed there in the lower layer (2.5 units, and 2.6;
    against 1.0 and 1.1): the wet paint it ploughs that can't get out is
    left where it was, under the new paint."""
    added, below = interior(120, True)
    assert added > 4.3
    assert below > 1.8


@pytest.mark.parametrize("bed", [True, False])
def test_a_wide_brush_s_set_down_bead_is_no_wall(bed):
    """Where a 120 px flat is set down, it squeezes not much more out behind
    it, for its size, than a 36 px flat does: its bead stands no more than
    3.3 times as high (2.5 and 2.7 down the bed, seeds 1 to 5 and 6 to 10;
    2.7 and 3.0 on bare canvas; it stood 5.9 to 7.1 times as high, a wall
    against the light), and no more than 1.8 times its side ridges (0.9 to
    1.3 times now)."""
    b120, b36 = bead(120, bed), bead(36, bed)
    assert b120 < 3.3 * b36
    assert b120 < 1.8 * ridges(120, bed)[0]


@pytest.mark.parametrize("bed", [True, False])
@pytest.mark.parametrize("bt", [brushes.FLAT_HOG, brushes.ROUND_HOG])
def test_a_wide_stroke_conserves_paint(bt, bed):
    """Every channel of the paint on the canvas and the brush together is
    the same after a hard 120 px stroke as before, down the bed or on bare
    canvas: what it holds back stays on it or under it. (On a canvas with
    room all round for what the stroke pushes ahead of it and its bead.)"""
    c = _canvas(400, bed, height=420)
    brush = Brush(bt.sized(120)).load(P.TITANIUM_WHITE, 150)
    start = _totals(c, brush)
    c.stroke([(200, 120), (200, 260)], None, brush=brush, pressure=0.9, seed=1)
    np.testing.assert_allclose(_totals(c, brush), start, rtol=1e-6)


@pytest.mark.parametrize("bed", [True, False])
def test_a_wide_stroke_off_the_edge_loses_only_what_goes_off(bed):
    """Run half off the canvas, a 120 px flat leaves at most what there was
    on the canvas and the brush, channel by channel, and none negative."""
    c = _canvas(240, bed)
    brush = Brush(brushes.FLAT_HOG.sized(120)).load(P.TITANIUM_WHITE, 150)
    start = _totals(c, brush)
    c.stroke([(10, Y0), (10, Y1)], None, brush=brush, pressure=0.9, seed=3)
    after = _totals(c, brush)
    assert np.all(after <= start * (1 + 1e-9) + 1e-9)
    assert after[IV] < start[IV]
    assert c.top.min() > -1e-5 and c.und.min() > -1e-5


def test_other_strokes_paint_beyond_its_reach_is_untouched():
    """A hard 120 px stroke down the bed leaves the bed exactly as it was
    beyond the furthest its ridges may reach (its room). Under its middle,
    the bed's paint it couldn't push aside stays in the bed's layer, and
    stays the bed's colour: its paint per unit differs from the bed's by
    0.19 of it (the median over the pixels; stirred a little into the new
    paint above, as it is everywhere a brush passes), against 0.23 when it
    was ploughed out and what came back was the stroke's."""
    c, _, x, _, _ = _stroke(120, True)
    room = 0.12 * 120 * (1 + brushes.FLAT_HOG.spread) + 1.0
    reach = 60 * 1.1 + room + 3
    far = np.abs(np.arange(c.width) + 0.5 - x) > reach
    bed = _canvas(c.width, True)
    assert np.array_equal(c.und[:, far], bed.und[:, far])
    assert np.array_equal(c.top[:, far], bed.top[:, far])
    u = c.und[ROWS, int(x - 30) : int(x + 31)].astype(np.float64)
    b0 = bed.und[0, 0].astype(np.float64)
    b0 = b0[:IV] / b0[IV]
    held = u[..., IV] > 0.5
    assert held.mean() > 0.95
    per_unit = u[held][:, :IV] / u[held][:, IV : IV + 1]
    off = np.abs(per_unit - b0).sum(axis=1) / np.abs(b0).sum()
    assert np.median(off) < 0.21


_LEVEL = {}


def _level_edge(w, seed):
    """A heavily loaded red stroke of a ``w`` px flat on bare canvas, level
    (stroke ``seed``): the share of its outer pixels holding over a unit
    that are more than 4 levels paler than its full paint (unlit luma)."""
    if (w, seed) not in _LEVEL:
        s = 2 * w + 100
        c = Canvas(s, s, seed=2)
        c.stroke(
            [(20, s / 2), (s - 20, s / 2)],
            P.CADMIUM_RED,
            amount=120,
            brush=brushes.FLAT_HOG.sized(w),
            seed=seed,
        )
        lum = 255.0 * (c.render(lighting=False) @ np.array([0.2126, 0.7152, 0.0722]))
        dark = lum[2, 2] - lum
        full = dark[s // 2, 60 : s - 60].mean()
        v = c.top[..., IV]
        edge = np.zeros(v.shape, bool)
        edge[:, 60 : s - 60] = True
        empty = v < 1e-3
        outer = (np.roll(empty, 1, 0) | np.roll(empty, -1, 0)) & ~empty & edge
        thick = outer & (v > 1.0)
        # (enough to measure: 45 to 202 at 70 px over seeds 1 to 10)
        assert thick.sum() > 30
        _LEVEL[w, seed] = float((full - dark[thick] > 4).mean())
    return _LEVEL[w, seed]


def _pale(w, seeds=SEEDS):
    """The median over ``seeds`` of _level_edge (a stroke now and then bulges
    at its edge and comes out far paler: seed 7 at 70 px, 69%, 71%
    before)."""
    return float(np.median([_level_edge(w, seed) for seed in seeds]))


def test_the_level_edge_of_heavy_paint_stays_solid_for_a_70_px_brush():
    """As test_the_level_edge_of_heavy_paint_stays_solid, for a 70 px flat:
    of the outer pixels of a heavy, level stroke holding over a unit, few
    are more than 4 levels paler than its paint: at most 8%, the median over
    ten strokes (seeds 1 to 10: 5.8%; 9.7% before. Over five the median
    is too spread to tell: 3.8% over seeds 1 to 5, 8.3% over 6 to 10,
    against 8.7% and 34% before)."""
    assert _pale(70, range(1, 11)) <= 0.08


@pytest.mark.xfail(reason="a 120 px flat's level edge still has pale feet", strict=True)
def test_the_level_edge_of_heavy_paint_stays_solid_for_a_120_px_brush():
    """The same for a 120 px flat, at the bound of the original test (2%): it
    does not hold (13%, the median over seeds 1 to 5): see the bound
    below."""
    assert _pale(120) <= 0.02


def test_the_level_edge_of_a_120_px_brush_is_bounded_pale():
    """A bound on what the previous test finds for a 120 px flat: no more
    than 20% of its outer pixels holding over a unit stay more than 4
    levels paler than its paint (the median over seeds 1 to 5: 13%, and 12%
    over seeds 6 to 10; 28% and 34% before), some by over 100 levels, as
    before: at the foot of a slope that drops more steeply in its last
    pixel than the slope above it says, or along a stretch of edge whose
    paint climbs from pixel to pixel (see _engine.RAMP_EVEN). (Covering
    such feet fully would undo the anti-aliasing of a slanting edge.)"""
    assert _pale(120) <= 0.2


@pytest.mark.parametrize(
    ("w", "amount", "least"),
    [(70, 60, 235), (120, 60, 235), (50, 120, 185), (120, 120, 230)],
)
def test_a_wide_stroke_s_edges_are_anti_aliased(w, amount, least):
    """As test_stroke_edges_are_anti_aliased, for 50 to 120 px strokes at a
    slant: most columns crossing their edge have an in-between pixel. The
    mean over five canvases (seeds 1 to 5, and 6 to 10): 240 (240) of 240
    at 60 units, 70 and 120 px (240 and 239, 240, before); heavily loaded
    (120 units), 199 (209) at 50 px (185, 195 before), and 240 (240) at
    120 px (233, 199 before)."""
    counts = []
    for seed in SEEDS:
        c = Canvas(400, 300, seed=seed, texture=0)
        c.stroke([(20, 90), (380, 210)], paint="#000000", amount=amount, width=w)
        img = c.render(lighting=False).mean(-1)
        ground = img[2, 2]
        lo, hi = img.min() + 0.05, ground - 0.05
        counts.append(
            sum(
                1 for x in range(80, 320) if ((img[:, x] > lo) & (img[:, x] < hi)).any()
            )
        )
    assert np.mean(counts) > least


def test_the_limit_starts_where_a_footprint_is_two_edge_free_wide():
    """Paint within EDGE_FREE px of the nearer edge of a footprint gets out
    freely; deeper, a share that falls continuously from 1 at EDGE_FREE."""
    f = eng.EDGE_FREE
    assert eng.escapes(0.0) == 1.0 and eng.escapes(f) == 1.0
    assert 0.99 < eng.escapes(f + 0.1 * eng.EDGE_SQUEEZE) < 1.0
    assert eng.escapes(f + eng.EDGE_SQUEEZE) == pytest.approx(0.5)
    assert eng.escapes(f + 20.0) < 0.03


# ----------------------------------------------------------------------------
# the mechanism, step by step


def _step(w=120, angle=30.0, i=2, bed=False):
    """Step ``i`` of a hard 120 px flat stroke turned ``angle`` degrees to
    its path (so that its footprint is narrower across the stroke than the
    brush, and its columns closer together): (run, step)."""
    c = _canvas(400, bed)
    stroke = Stroke(
        [(200, 100), (200, 250)],
        P.TITANIUM_WHITE,
        amount=150,
        brush=brushes.FLAT_HOG.sized(w),
        pressure=0.9,
        seed=1,
        angle=angle,
        follow_path=True,
    )
    run = sm._Run.begin(c, stroke)
    got = run.step(i)
    assert got is not None
    return run, got[0]


def test_squeezed_paint_gets_out_by_how_deep_its_column_lies():
    """Each column of a wide brush squeezes out the share of its overload
    that escapes from as deep as it lies under the footprint: half the
    footprint's width across the stroke, less how far across the stroke
    the column lies (|u| r / |uscale|; on a flat turned 30 degrees, closer
    in than its place across the bristles says). Measured one column at a
    time, against the column that squeezes out most (one within EDGE_FREE of
    an edge)."""
    run, st = _step()
    half = st.hold.r * st.hold.lmax
    assert half > eng.EDGE_FREE + 10
    n = eng.COLUMNS
    out = np.zeros(n)
    for b in range(n):
        tip = np.zeros((n, eng.PACKED))
        belly = np.zeros((n, eng.PACKED))
        belly[:, :] = 1.0
        belly[:, IV] = 1e6
        br = eng.Load(tip=tip, belly=belly, heel=np.zeros((n, eng.PACKED)))
        load = np.zeros(n)
        load[b] = 50.0
        ridges = eng.Ridges(
            np.zeros(eng.PACKED), np.zeros(eng.PACKED), np.zeros(eng.PACKED)
        )
        eng.squeeze_out(
            br, run.K, st, belly[:, IV].copy(), load, 10.0, ridges, run.col_u
        )
        out[b] = ridges.left[IV] + ridges.right[IV] + ridges.ring[IV]
    depth = half - np.abs(run.col_u) * st.hold.r / abs(st.hold.uscale)
    want = np.array([eng.escapes(d) for d in depth])
    assert want.min() < 0.05 and want.max() == 1.0
    np.testing.assert_allclose(out / out.max(), want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("pick", [0.1, 0.25, 0.5])
def test_only_what_goes_aside_is_held_back(pick):
    """Of the wet paint a wide brush's bristles would shift at a pixel, the
    brush picks up its share as anywhere, what would go ahead goes ahead,
    and of what would go aside only the share that escapes from that deep
    does: the rest is not shifted at all (_held_back, per pixel)."""
    half = 60.0
    d_in = np.array([5.0, 20.0, 20.5, 21.0, 23.0, 30.0, 50.0])
    lat = (half - d_in)[None, :]
    for s in (0.39, 0.7, 1.0):
        E = np.ones((1, d_in.size))
        side = np.full((1, d_in.size), s)
        picks = eng._held_back.py_func(E, side, lat, half, pick)
        f = np.array([eng.escapes(d) for d in d_in])
        brush = picks[0] * E[0]
        rest = E[0] - brush
        np.testing.assert_allclose(brush, pick, rtol=1e-12)
        np.testing.assert_allclose(
            side[0] * rest, s * f * (1 - pick), rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(
            rest * (1 - side[0]), (1 - s) * (1 - pick), atol=1e-12
        )
        np.testing.assert_allclose(1 - E[0], s * (1 - f) * (1 - pick), atol=1e-12)
        if s == 1.0:
            # all that is shifted goes aside: exactly, so none goes ahead
            assert np.all(side[0] == 1.0)


def test_paint_shifted_where_a_wide_brush_is_set_down_all_goes_into_the_bead():
    """As a wide brush is set down, all the wet paint it shifts (and doesn't
    pick up) goes all round into the bead: none to its sides, none lost."""
    run, st = _step(i=1, angle=0.0, bed=True)
    assert st.motion.touchdown
    win = st.win
    moved = np.ones((win.ch, win.cw, eng.PACKED))
    lat = np.linspace(-80, 80, win.cw)[None, :].repeat(win.ch, 0)
    side = np.full((win.ch, win.cw), 0.3)
    ridges = eng.Ridges(
        np.zeros(eng.PACKED), np.zeros(eng.PACKED), np.zeros(eng.PACKED)
    )
    eng._to_sides_or_ahead(run.cv, run.pend, st, run.K, ridges, moved, lat, side)
    assert ridges.left[IV] == 0 and ridges.right[IV] == 0
    assert ridges.ring[IV] == pytest.approx(win.ch * win.cw, rel=1e-12)


def test_a_clean_wide_brush_picks_up_and_leaves_the_rest_under_it():
    """A clean 120 px flat dragged hard down the bed picks up at least as much
    as when it threw what it couldn't get out of the way aside (21 000
    units now, 16 000 before; 14 600 when the brush's share was taken only
    of what it shifted): it meets again what stays under it. Under its
    middle the bed stays in the lower layer, 2.3 units of it (1.0
    before)."""
    c = _canvas(400, True)
    brush = Brush(brushes.FLAT_HOG.sized(120))
    c.stroke([(200, Y0), (200, Y1)], None, brush=brush, pressure=0.9, seed=1)
    assert brush.volume > 18000
    assert c.und_v[ROWS, 170:231].mean() > 1.8


def _ramp(rows, ids=None, a_foot=0.36, a_in=1.0, W=12, foot=None):
    """A level edge of the stroke 1's paint on top: the volumes ``rows``
    (inner rows first), the same all along; the inner rows covered ``a_in``,
    the last ``a_foot``; then two empty rows. (top, top_a, ids)."""
    H = len(rows) + 2
    top = np.zeros((H, W, eng.PACKED), np.float32)
    top_a = np.zeros((H, W), np.float32)
    own = np.zeros((H, W), np.int32)
    for k, v in enumerate(rows):
        top[k, :, :] = 0.5 * v
        top[k, :, IV] = v
        top_a[k, :] = a_in if k < len(rows) - 1 else a_foot
        own[k, :] = 1
    if ids is not None:
        own[len(rows) - 1, :] = ids
    if foot is not None:
        foot(top, top_a, own)
    return top, top_a, own


def _cover(top, top_a, own, ramps=1.0):
    a = top_a.copy()
    eng.cover_ramps(top, a, own, 1, (0, top.shape[0], 0, top.shape[1]), ramps)
    return a


def test_the_foot_of_a_slope_covers_as_the_slope_says():
    """The last pixel of a stroke's own paint at the foot of a slope (6, 4.4,
    3 and then 2) covers at least as much of itself as the slope carried on
    says: all of it; with ``ramps`` 0.5, at least half; and a second pass
    changes nothing (a pixel it covered more never says how the slope
    runs)."""
    top, top_a, own = _ramp([6.0, 4.4, 3.0, 2.0])
    once = _cover(top, top_a, own)
    assert np.all(once[3, 2:-2] == 1.0)
    half = _cover(top, top_a, own, 0.5)
    np.testing.assert_allclose(half[3, 2:-2], 0.5, rtol=1e-6)
    np.testing.assert_array_equal(_cover(top, once, own), once)


def test_the_slope_cover_leaves_other_strokes_and_thin_traces():
    """It covers only the stroke's own paint: the foot of another stroke's
    paint keeps its coverage; so do pixels whose inner neighbours are only
    partly covered (they say nothing about the slope). So does a trace
    thinner than MIN_FILM at the foot (0.14 units, covering 1% of its
    pixel). (The MIN_FILM floor in cover_ramps is a stated guard: such a
    trace makes every pixel beside it an edge pixel, which is never read
    as the slope, so it would be left without the floor too.)"""
    top, top_a, own = _ramp([6.0, 4.4, 3.0, 2.0], ids=2)
    np.testing.assert_array_equal(_cover(top, top_a, own), top_a)
    top, top_a, own = _ramp([6.0, 4.4, 3.0, 2.0], a_in=0.6)
    np.testing.assert_array_equal(_cover(top, top_a, own), top_a)

    def trace(top, top_a, own):
        top[3, 5, :] = 0.07
        top[3, 5, IV] = 0.14
        top_a[3, 5] = 0.01

    top, top_a, own = _ramp([6.0, 4.4, 3.0, 2.0], foot=trace)
    a = _cover(top, top_a, own)
    assert a[3, 5] == top_a[3, 5]
    assert a[3, 2] == 1.0


def test_the_slope_cover_leaves_a_plateau_and_a_staircase():
    """Where the paint inward of the foot hardly rises (3.2 then 3.0: a
    plateau's edge, a rise of 6%), and where the foot is a step of a slanting
    edge's staircase (a neighbour along the edge holds nothing), the foot
    keeps its anti-aliasing."""
    top, top_a, own = _ramp([3.4, 3.2, 3.0, 2.0])
    np.testing.assert_array_equal(_cover(top, top_a, own), top_a)

    def step(top, top_a, own):
        top[3, 6:] = 0.0
        top_a[3, 6:] = 0.0

    top, top_a, own = _ramp([6.0, 4.4, 3.0, 2.0], foot=step)
    a = _cover(top, top_a, own)
    assert a[3, 5] == top_a[3, 5]
    assert a[3, 3] == 1.0


@pytest.mark.parametrize(("left", "covered"), [(1.9, True), (1.6, False)])
def test_the_slope_cover_leaves_a_foot_between_unlike_neighbours(left, covered):
    """A foot whose neighbours along the edge hold much alike (2.0 and 1.9:
    5% apart) is on a run of a level edge and is covered as the slope says;
    one between 2.0 and 1.6 (20% apart, more than RAMP_EVEN) is where a
    slanting edge's staircase climbs from pixel to pixel, and keeps its
    anti-aliasing."""

    def uneven(top, top_a, own):
        top[3, 4, IV] = left

    top, top_a, own = _ramp([6.0, 4.4, 3.0, 2.0], foot=uneven)
    a = _cover(top, top_a, own)
    assert (a[3, 5] == 1.0) == covered
    assert a[3, 5] in (1.0, top_a[3, 5])


def test_the_slope_cover_goes_by_the_footprint_s_width_across_the_stroke():
    """Whether a stroke's edges are covered as their slope says goes by how
    wide its footprint reached across the stroke (the most of r * lmax over
    its steps), not by the brush's size: a 120 px flat dragged edge-on
    reaches 14 px either side of its centre, not at all; flat on, 63."""
    for angle, lo, hi in ((90.0, 10.0, eng.RAMP_FROM), (0.0, 60.0, 66.0)):
        stroke = Stroke(
            [(200, 100), (200, 250)],
            P.TITANIUM_WHITE,
            amount=20,
            brush=brushes.FLAT_HOG.sized(120),
            pressure=0.9,
            seed=1,
            angle=angle,
            follow_path=True,
        )
        run = sm._Run.begin(_canvas(400, False), stroke)
        for _ in run.steps():
            pass
        assert lo < run.lateral < hi


def test_covering_a_real_stroke_s_feet_again_changes_nothing():
    """Once a heavily loaded 120 px stroke has settled, covering the feet of
    its slopes again changes no pixel's coverage (a pixel it covered more
    never says how the slope runs: no cascade, stroke after stroke)."""
    c = Canvas(340, 340, seed=2)
    c.stroke(
        [(20, 170), (320, 170)],
        P.CADMIUM_RED,
        amount=120,
        brush=brushes.FLAT_HOG.sized(120),
        seed=1,
    )
    sid = int(c.top_id.max())
    before = c.top_a.copy()
    for _ in range(3):
        eng.cover_ramps(c.top, c.top_a, c.top_id, sid, (0, 340, 0, 340), 1.0)
    np.testing.assert_array_equal(c.top_a, before)


@pytest.mark.parametrize(
    ("w", "angle", "ramps"), [(120, 90.0, None), (120, 0.0, 1.0), (32, 0.0, 0.447)]
)
def test_the_slope_cover_is_not_run_for_a_brush_dragged_edge_on(
    monkeypatch, w, angle, ramps
):
    """A 120 px flat dragged edge-on paints a mark 28 px wide: its edges are
    not covered as their slope says (the brush's size doesn't count, how
    wide it paints does); flat on, they are, fully. In between, as a 32 px
    flat flat on (its footprint reaching 16.8 px either side of its centre,
    between RAMP_FROM and EDGE_FREE), they are covered only in part:
    ``ramps`` blends from none to all."""
    calls = []
    monkeypatch.setattr(eng, "cover_ramps", lambda *a: calls.append(a[-1]))
    c = _canvas(400, False)
    c.stroke(
        [(200, 100), (200, 250)],
        P.TITANIUM_WHITE,
        amount=60,
        brush=brushes.FLAT_HOG.sized(w),
        pressure=0.9,
        seed=1,
        angle=angle,
        follow_path=True,
    )
    if ramps is None:
        assert calls == []
    else:
        assert len(calls) == 1
        assert calls[0] == pytest.approx(ramps, abs=0.005)
