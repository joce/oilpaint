"""A brush pressed hard through thick wet paint ploughs it: its bristles go
down through the paint they carry, cut the old paint away (deeper along the
lines of their clumps), push it up beside the stroke, lay their own paint
more into their grooves, and the old paint folds up over the new between
them: wet-in-wet marbling (see _engine.GAP). The stiffer the hair, the more.
Pressed lightly, or through a thin film, a brush brushes as it always did.

The measures are averaged over a few strokes (seeds) and taken on the paint
where they can be: one stroke's hair pattern is chance."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, Stroke, brushes, spline
from oilpaint import _engine as eng
from oilpaint import palette as P
from oilpaint import stroke as stm
from oilpaint._engine import IB, IS, IV

W, H = 200, 240
X = W // 2  # where the stroke crosses the bed
ROWS = slice(50, H - 50)
LUM = np.array([0.2126, 0.7152, 0.0722])
SEEDS = (1, 2, 3)

_PIGMENTS = np.stack(
    [np.concatenate([p.K, [p.S, 1.0]]) for p in (P.ULTRAMARINE_BLUE, P.TITANIUM_WHITE)],
    axis=1,
)
_SCALE = 1.0 / np.abs(_PIGMENTS).max(axis=1)


def _split(layer):
    """The volume of (ultramarine, white) at each pixel of a packed layer."""
    a = layer[..., : IV + 1].astype(np.float64) * _SCALE
    sol, *_ = np.linalg.lstsq(
        _PIGMENTS * _SCALE[:, None], a.reshape(-1, IV + 1).T, rcond=None
    )
    return sol.T.reshape(layer.shape[:2] + (2,))


_BEDS = {}


def _bed(thickness=5.0, width=W):
    """A canvas covered with wet ultramarine about ``thickness`` units thick
    (laid lightly, then thickened, so the bed doesn't depend on ploughing)."""
    key = (thickness, width)
    if key not in _BEDS:
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
        scale = np.float32(thickness / v)
        c.top *= scale
        c.und *= scale
        _BEDS[key] = c
    return _BEDS[key].copy()


class _Finer(spline.Path):
    """A path sampled ``k`` times as finely as a stroke asks."""

    def __init__(self, points, k):
        self.path, self.k = spline.as_path(points), k
        self.n_knots = self.path.n_knots

    def _dense(self):
        return self.path._dense()

    def sample(self, spacing):
        return self.path.sample(spacing / self.k)


def _cross(bt, pressure, seed, c=None, k=None, paint=True):
    """A brush of type ``bt`` crosses the bed top to bottom, loaded with white
    (or clean). Returns (the share of the blue left in the middle third of
    the stroke, the highest blue beside it against the bed, how much the
    share of blue in the visible layer varies across the stroke: the
    streaks) and the canvas."""
    c = _bed() if c is None else c
    before = (_split(c.top) + _split(c.und))[..., 0]
    path = [(X, 5), (X, H - 5)]
    c.stroke(
        _Finer(path, k) if k else path,
        P.TITANIUM_WHITE if paint else None,
        amount=45 * bt.width / 36,
        brush=bt,
        pressure=pressure,
        seed=seed,
    )
    top = _split(c.top)
    after = top[..., 0] + _split(c.und)[..., 0]
    r = int(bt.width)
    core = np.s_[ROWS, X - r // 3 : X + r // 3 + 1]
    left = after[core].sum() / before[core].sum()
    profile = after[ROWS, X - 2 * r : X + 2 * r + 1].mean(axis=0)
    bed = before[ROWS, X - 2 * r : X + 2 * r + 1].mean()
    lo, hi = int(0.4 * r), int(0.9 * r) + 1
    beside = profile[2 * r - hi : 2 * r - lo + 1].max()
    beside += profile[2 * r + lo : 2 * r + hi].max()
    share = top[..., 0][core] / np.maximum(top[core].sum(axis=-1), 1e-6)
    return np.array([left, 0.5 * beside / bed, share.std(axis=1).mean()]), c


def _mean_cross(bt, pressure, **kw):
    return np.mean([_cross(bt, pressure, s, **kw)[0] for s in SEEDS], axis=0)


HOG = brushes.FLAT_HOG.sized(36)


def test_a_hard_stroke_ploughs_and_marbles_thick_wet_paint():
    """White pressed hard through about 5 units of wet blue clears about
    half of it (nine tenths stayed when the brush rode on its load), heaps
    it up beside the stroke and streaks the stroke with it."""
    left, ridge, streaks = _mean_cross(HOG, 0.9)
    assert left < 0.6
    assert ridge > 1.6
    assert streaks > 0.17


def test_the_streaks_hold_tints_not_film_or_bare():
    """The rendered stroke is streaked (a good share of it is darker than
    its film), and the streaks are tints of the two paints, not the film
    and the bare blue: of the pixels darker than the film, most lie between
    the two."""
    darker, between = [], []
    for s in SEEDS:
        c = _bed()
        bed = np.median(c.render(lighting=False)[ROWS, 10:30] @ LUM)
        _, c = _cross(HOG, 0.9, s, c=c)
        lum = c.render(lighting=False)[ROWS, X - 12 : X + 13] @ LUM
        film = np.median(lum)
        dark = lum < bed + 0.75 * (film - bed)
        darker.append(dark.mean())
        between.append((dark & (lum > bed + 0.25 * (film - bed))).sum() / dark.sum())
    assert np.mean(darker) > 0.15
    assert np.mean(between) > 0.45


def test_stiffer_hair_ploughs_more():
    """From soft hair to stiff, a brush clears more of the old paint and
    streaks the stroke more (no window lets one kind through and not the
    next); near the stiff end it levels off, without falling back."""
    sweep = {
        s: _mean_cross(brushes.FLAT_HOG.sized(36).with_(stiffness=s), 0.8)
        for s in (0.2, 0.5, 0.8, 1.0)
    }
    left = {s: r[0] for s, r in sweep.items()}
    streaks = {s: r[2] for s, r in sweep.items()}
    assert left[0.2] > left[0.5] + 0.08
    assert left[0.5] > left[1.0] + 0.2
    assert streaks[0.2] < streaks[0.5] - 0.04
    assert streaks[0.5] < streaks[1.0] - 0.1
    assert left[1.0] < left[0.8] + 0.03
    assert streaks[1.0] > streaks[0.8] - 0.03


def test_brushes_plough_as_stiff_as_their_hair():
    """At the same pressure a sable ploughs a little, a synthetic round more,
    a hog flat much more, and a bright (stiffer still) at least as much."""
    results = {
        name: _mean_cross(getattr(brushes, name).sized(size), 0.8)
        for name, size in (
            ("FLAT_SABLE", 36),
            ("ROUND", 24),
            ("FLAT_HOG", 36),
            ("BRIGHT", 36),
        )
    }
    left = {k: v[0] for k, v in results.items()}
    streaks = {k: v[2] for k, v in results.items()}
    assert left["FLAT_SABLE"] > left["ROUND"] + 0.08
    assert left["ROUND"] > left["FLAT_HOG"] + 0.12
    assert left["BRIGHT"] < left["FLAT_HOG"]
    assert streaks["FLAT_SABLE"] < streaks["ROUND"] < streaks["FLAT_HOG"]
    # a sable isn't shut out: it still clears a little more than a light touch
    assert left["FLAT_SABLE"] < _mean_cross(HOG, 0.3)[0]


def _ploughing_during(c, stroke):
    """How far the brush ploughs (see _engine._ploughs) at its most at each
    step of ``stroke``, painted on ``c`` step by step."""
    run = stm._Run.begin(c, stroke)
    y0, y1, x0, x1 = run.bounds()
    run.pend = stm._pending(y0, y1, x0, x1)
    out = []
    for st, tables in run.steps():
        F, _ = eng.footprint(st, run.K, run.tex, run.tex2, *tables)
        w, _, _ = eng._ploughs(run.cv, F, st, run.K, run.pend)
        out.append(w.max())
        stm._brush_step(run, st, tables)
    return np.array(out)


def _white(pressure, brush=HOG, **kw):
    return Stroke(
        path=[(X, 5), (X, H - 5)],
        paint=P.TITANIUM_WHITE,
        amount=45,
        brush=brush,
        pressure=pressure,
        seed=3,
        **kw,
    )


def test_it_ploughs_only_pressed_hard_through_thick_stiff_paint():
    """During a stroke, it ploughs nowhere at any step (and so takes the
    steps it always took) when pressed at 0.4 or less, over wet paint under
    about 1.5 units thick (its own paint stirred in doesn't count), over
    paint thinned for wet-on-wet work, or over bare canvas; nor does a
    dab."""
    assert _ploughing_during(_bed(), _white(0.9)).max() > 0.9
    for p in (0.1, 0.3, 0.4):
        assert _ploughing_during(_bed(), _white(p)).max() == 0
    assert _ploughing_during(_bed(0.8), _white(0.9)).max() == 0  # (at most 1.4)
    assert _ploughing_during(Canvas(W, H, seed=5), _white(0.9)).max() == 0
    soft = _bed()
    soft.top[..., IB] *= 0.5
    soft.und[..., IB] *= 0.5
    assert _ploughing_during(soft, _white(0.9)).max() == 0
    dab = Stroke(path=[(X, 100)], paint=P.TITANIUM_WHITE, amount=45, pressure=0.9)
    assert _ploughing_during(_bed(), dab).max() == 0


def test_it_ploughs_where_there_is_thick_paint_and_nowhere_else():
    """Half on thick paint and half on bare canvas, a stroke lays on the bare
    half what it would on a bare canvas; and a narrow, thick ridge of paint
    under the middle of a wide brush is ploughed."""
    cols = np.s_[ROWS, X + 3 : X + 18]
    straddle, bare = [], []
    for s in SEEDS:
        c = _bed(8.0)
        c.top[:, X:] = 0
        c.und[:, X:] = 0
        c.top_a[:, X:] = 0
        c.und_a[:, X:] = 0
        b = Canvas(W, H, seed=5)
        for cc in (c, b):
            cc.stroke(
                [(X, 5), (X, H - 5)], P.TITANIUM_WHITE, amount=45, brush=HOG,
                pressure=0.9, seed=s,
            )  # fmt: skip
        straddle.append((c.top_v + c.und_v)[cols].mean(axis=0))
        bare.append((b.top_v + b.und_v)[cols].mean(axis=0))
    rel = np.mean(straddle, axis=0) / np.mean(bare, axis=0) - 1
    assert np.abs(rel).max() < 0.05
    # a ridge 10 px wide and 6 units thick
    left = []
    for pressure in (0.3, 0.9):
        c = Canvas(W, H, seed=5)
        c.stroke(
            [(X, -10), (X, H + 10)], P.ULTRAMARINE_BLUE, amount=60,
            brush=brushes.FLAT_HOG.sized(10), pressure=0.1, seed=0,
        )  # fmt: skip
        v = float((c.top_v + c.und_v)[ROWS, X - 5 : X + 5].mean())
        c.top *= np.float32(6.0 / v)
        c.und *= np.float32(6.0 / v)
        before = (_split(c.top) + _split(c.und))[..., 0][ROWS].sum()
        _, c = _cross(HOG, pressure, 1, c=c)
        near = (_split(c.top) + _split(c.und))[..., 0][ROWS, X - 8 : X + 8]
        left.append(near.sum() / before)
    assert left[1] < 0.65 * left[0]


def test_ploughing_lays_as_much_as_it_would_have():
    """Where a brush ploughs it shifts what it lays towards its grooves, but
    lays as much in all."""
    rng = np.random.default_rng(0)
    K = eng.StrokeConsts(*([0] * len(eng.StrokeConsts._fields)))
    K = K._replace(streaks=0.55, stiffness=0.8)
    win = eng.Window(x0=0, y0=0, hh=20, ww=30, cy0=0, cx0=0, ch=20, cw=30, wy0=0, wx0=0)
    part = win._replace(cy0=3, cx0=4, ch=12, cw=20, wy0=3, wx0=4)
    plough = rng.uniform(0, 1, (20, 30))
    lane = np.exp(rng.uniform(-3, 3, (20, 30)))
    off = np.array([[0, 0, 3, 4], [1, 2, 4, 5]], np.int64)
    d = rng.uniform(0, 2, (12, 20))
    d_off = rng.uniform(0, 2, 2)
    total = d.sum() + d_off.sum()
    d0 = d.copy()
    eng._into_grooves(d, d_off, win, part, off, K, (plough, True, lane))
    assert d.sum() + d_off.sum() == pytest.approx(total, rel=1e-12)
    assert not np.allclose(d, d0)
    assert d.min() > 0


def _totals(c, brush):
    on_canvas = c.top.sum(axis=(0, 1), dtype=np.float64)
    on_canvas += c.und.sum(axis=(0, 1), dtype=np.float64)
    return on_canvas + (brush.tip + brush.reservoir + brush.heel).sum(axis=0)


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ([(X, 40), (X, 200)], "FLAT_HOG"),
        ([(50, 40), (150, 90), (60, 140), (150, 200)], "BRIGHT"),  # turning
        ([(150, 200), (X, 120), (150, 40)], "ROUND"),
    ],
)
def test_ploughing_keeps_every_pigment(path, name):
    """Every band of K, S, volume and body: what is on the canvas and on the
    brush after a ploughing stroke (well inside the canvas: nothing it
    pushes reaches the edge) is what there was before."""
    c = _bed()
    brush = Brush(getattr(brushes, name).sized(30)).load(P.TITANIUM_WHITE, 40)
    before = _totals(c, brush)
    c.stroke(path, brush=brush, pressure=0.9, seed=3)
    np.testing.assert_allclose(_totals(c, brush), before, rtol=1e-6, atol=0)
    assert c.top.min() > -1e-5 and c.und.min() > -1e-5


def test_ploughing_off_the_edge_of_the_canvas_creates_no_paint():
    """A ploughing stroke that runs off the canvas leaves at most what there
    was, pigment by pigment (the rest went off the canvas), and no channel
    goes negative."""
    c = _bed()
    brush = Brush(HOG).load(P.TITANIUM_WHITE, 45)
    before = _totals(c, brush)
    c.stroke([(120, 100), (300, 110)], brush=brush, pressure=0.9, seed=3)
    after = _totals(c, brush)
    assert np.all(after <= before * (1 + 1e-6) + 1e-9)
    assert after[IV] < before[IV] - 1.0  # some went off the canvas
    assert after[IS] < before[IS]
    assert c.top.min() > -1e-5 and c.und.min() > -1e-5


def test_ploughing_does_not_depend_on_how_finely_the_stroke_is_cut():
    """Sampled twice as finely, a stroke clears about as much, heaps about as
    much beside it and streaks about as much (on average over a few strokes:
    a finer sampling draws a different hair pattern)."""
    coarse = _mean_cross(HOG, 0.9)
    fine = _mean_cross(HOG, 0.9, k=2)
    assert abs(fine[0] - coarse[0]) < 0.08
    assert abs(fine[1] - coarse[1]) < 0.3
    assert abs(fine[2] - coarse[2]) < 0.1


def test_ploughing_repeats_bit_for_bit():
    def run():
        c = _bed()
        c.stroke(
            [(X, 5), (X + 30, 120), (X, H - 5)],
            P.TITANIUM_WHITE,
            amount=45,
            brush=HOG,
            pressure=0.9,
            seed=4,
        )
        return c

    a, b = run(), run()
    for name in ("top", "und", "top_a", "und_a", "top_id"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_a_step_wholly_off_the_canvas_reads_no_ploughing():
    """A step whose window lies wholly off the canvas has no ploughing to
    read there (its pixels are taken as at the canvas's edge, outside the
    window's part on the canvas): it lays as it always did."""
    empty = np.zeros((0, 7))
    assert eng._at(empty, 0, 3, 0.0) == 0.0
    assert eng._at(np.ones((2, 2)), 2, 0, 1.5) == 1.5
    assert eng._at(np.full((2, 2), 0.25), 1, 1, 0.0) == 0.25
    # a stroke coming in from off a bare canvas, pressed hard, lays as one
    # pressed as hard coming in from off a canvas with paint far from it
    laid = []
    for far in (False, True):
        c = Canvas(W, H, seed=5)
        if far:
            c.stroke(
                [(150, 20), (190, 20)], P.ULTRAMARINE_BLUE, amount=60,
                brush=brushes.FLAT_HOG.sized(20), pressure=0.1, seed=1,
            )  # fmt: skip
        before = float((c.top_v + c.und_v).sum())
        c.stroke(
            [(-60, 150), (120, 150)], P.TITANIUM_WHITE, amount=45, brush=HOG,
            pressure=0.8, seed=2,
        )  # fmt: skip
        laid.append(float((c.top_v + c.und_v).sum()) - before)
    assert laid[0] == pytest.approx(laid[1], rel=1e-6)


def test_drops_of_spatter_do_not_plough(monkeypatch):
    """A drop of paint has no bristles: landing on thick, stiff wet paint, at
    a slant (laid as a short stroke, pressed at 0.6) or not, it lies on top
    and ploughs nothing at any step."""
    seen = []
    step = stm._brush_step

    def watched(run, st, tables):
        F, _ = eng.footprint(st, run.K, run.tex, run.tex2, *tables)
        seen.append(eng._ploughs(run.cv, F, st, run.K, run.pend)[1])
        step(run, st, tables)

    monkeypatch.setattr(stm, "_brush_step", watched)
    c = _bed()
    c.spatter(
        [(20, 60), (60, 160)], P.TITANIUM_WHITE, amount=4, size=4, reach=80, seed=2
    )
    assert len(seen) > 20 and not any(seen)
    # (a brush pressed as hard there does)
    seen.clear()
    c.stroke([(X, 5), (X, H - 5)], P.TITANIUM_WHITE, amount=45, brush=HOG, pressure=0.6)
    assert any(seen)


@pytest.mark.parametrize("stiffness", [-1.0, 1.5])
def test_stiffness_out_of_range_keeps_every_pigment(stiffness):
    """A brush given a stiffness outside 0 to 1 is taken as the nearest end
    of it: pressed hard through thick paint, it neither makes nor loses
    paint."""
    c = _bed()
    brush = Brush(brushes.FLAT_HOG.sized(30).with_(stiffness=stiffness))
    before = _totals(c, brush)
    c.stroke([(X, 40), (X, 200)], brush=brush, pressure=0.95, seed=3)
    np.testing.assert_allclose(_totals(c, brush), before, rtol=1e-6, atol=0)
    assert c.top.min() > -1e-5 and c.und.min() > -1e-5


def test_a_brush_that_does_not_mix_ploughs_but_does_not_fold():
    """Folding the old paint up over the new is mixing between the layers: a
    brush with no mixing still cuts the paint away and pushes it aside, but
    folds nothing."""

    def folds_after(**kw):
        c = _bed()
        run = stm._Run.begin(c, _white(0.9, **kw))
        y0, y1, x0, x1 = run.bounds()
        run.pend = stm._pending(y0, y1, x0, x1)
        for st, tables in run.steps():
            stm._brush_step(run, st, tables)
        return run.pend.folds.max()

    assert folds_after() > 0.5
    assert folds_after(mixing=0.0) == 0
    left, ridge, _ = _mean_cross(brushes.FLAT_HOG.sized(36).with_(mixing=0.0), 0.9)
    assert left < 0.65
    assert ridge > 1.5


def test_the_ploughs_property_stays_out_of_a_brush_repr():
    """A brush type's repr seeds the pattern of tips its dabs print (see
    _dab): ``ploughs`` stays out of it, so dabs and stipples print as they
    always did."""
    assert "ploughs" not in repr(brushes.STIPPLER)
    assert not brushes.FLAT_HOG.with_(ploughs=False).ploughs
