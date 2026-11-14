"""Fixes in the compiled stroke engine: strokes cut by the edge of the
canvas, pigment kept by dabs of a multi-colour brush, and coverage that does
not hang on the sign of a speck of paint."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes
from oilpaint import palette as P
from oilpaint._engine import IV, MIN_FILM

# ----------------------------------------------------------------------------
# a stroke cut by the edge of the canvas

W, H, M = 400, 120, 80


def _edge_pair(bt, edge, share, width=None):
    """The same stroke, cut by ``edge`` of the canvas (``share`` of its width
    off it), and uncut on a larger canvas: the two thickness maps over the
    part of the canvas they share (strokes run left to right), the rows inside
    the mark, and the brushes after."""
    w = width or bt.width
    horiz = edge in ("bottom", "top")
    cw, ch = (W, H) if horiz else (H, W)
    if horiz:
        y = ch - w / 2 + share * w if edge == "bottom" else w / 2 - share * w
        path = [(30.0, y), (cw - 30.0, y)]
    else:
        x = cw - w / 2 + share * w if edge == "right" else w / 2 - share * w
        path = [(x, 20.0), (x, ch - 20.0)]
    kw = {"amount": 8, "brush": bt, "seed": 9}
    if width:
        kw["width"] = width
    cut = Canvas(cw, ch, texture=0, seed=0)
    big = Canvas(cw + 2 * M, ch + 2 * M, texture=0, seed=0)
    b_cut = cut.stroke(path, P.CADMIUM_RED, **kw)
    b_big = big.stroke([(px + M, py + M) for px, py in path], P.CADMIUM_RED, **kw)
    t_cut = cut.wet_thickness.astype(np.float64)
    t_big = big.wet_thickness[M : M + ch, M : M + cw].astype(np.float64)
    if not horiz:
        t_cut, t_big = t_cut.T, t_big.T
    c = path[0][1] if horiz else path[0][0]
    n = ch if horiz else cw
    if edge in ("bottom", "right"):
        rows = slice(int(c - w / 2) + 2, int(min(c + w / 2, n)) - 2)
    else:
        rows = slice(2, int(c + w / 2) - 2)
    return t_cut, t_big, rows, (cut, b_cut), (big, b_big)


def _painted_end(t, rows, thr=0.15):
    return int(np.nonzero(t[rows].mean(axis=0) > thr)[0].max())


@pytest.mark.parametrize(
    ("name", "edge", "share", "width"),
    [
        ("FLAT_HOG", "bottom", 0.3, None),
        ("ROUND", "left", 0.5, None),
        ("ROUND", "top", 0.3, 30),
        ("FILBERT_HOG", "right", 0.5, None),
    ],
)
def test_a_stroke_cut_by_the_edge_paints_its_visible_part_as_uncut(
    name, edge, share, width
):
    """The part of a stroke on the canvas is painted as the same part of the
    stroke uncut: the brush doesn't run dry early because a share of it is
    past the edge (it did, 20 to 50 px early on these strokes)."""
    bt = getattr(brushes, name)
    t_cut, t_big, rows, _, _ = _edge_pair(bt, edge, share, width)
    end_cut, end_big = _painted_end(t_cut, rows), _painted_end(t_big, rows)
    assert abs(end_cut - end_big) <= 10
    # over the body of the stroke (all but its last two widths) the paint is
    # the same, to well under a percent
    body = slice(0, end_big - 2 * int(width or bt.width))
    diff = np.abs(t_cut[:, body] - t_big[:, body]).sum()
    assert diff <= 5e-3 * t_big[:, body].sum()


def test_a_stroke_cut_by_the_edge_creates_no_paint():
    """What is on the canvas and left on the brush after a stroke that runs
    past the edge is at most what was loaded, pigment by pigment (the rest
    went off the canvas); and no channel goes negative. (A guard: this held
    before the fix too.)"""
    for name, edge, share in (("FLAT_HOG", "bottom", 0.5), ("ROUND_HOG", "left", 0.7)):
        _, _, _, (cut, brush), _ = _edge_pair(getattr(brushes, name), edge, share)
        on_canvas = cut.top.sum(axis=(0, 1), dtype=np.float64) + cut.und.sum(
            axis=(0, 1), dtype=np.float64
        )
        left = (brush.tip + brush.reservoir + brush.heel).sum(axis=0)
        full = Brush(getattr(brushes, name))
        full.load(P.CADMIUM_RED, 8)
        start = (full.tip + full.reservoir + full.heel).sum(axis=0)
        assert np.all(on_canvas + left <= start * (1 + 1e-6) + 1e-9)
        assert (on_canvas + left)[IV] < start[IV]  # some went off the canvas
        # (as on the larger canvas: a K band of a pixel left with no paint
        # can hold -4e-6, single-precision rounding)
        assert cut.top.min() > -1e-5 and cut.und.min() > -1e-5
        assert np.isfinite(cut.top).all() and np.isfinite(cut.und).all()


def test_a_stroke_inside_the_canvas_is_painted_the_same_wherever_it_is():
    """Inside the canvas nothing about its edge comes in: the same strokes, a
    dab and a knife stroke on a canvas and moved onto a larger one paint the
    same, bit for bit. (A guard: this held before the fix too.)"""

    def paint(c, dx, dy):
        c.stroke(
            [(60 + dx, 50 + dy), (140 + dx, 70 + dy), (220 + dx, 55 + dy)],
            P.ULTRAMARINE_BLUE,
            amount=12,
            brush=brushes.FLAT_HOG,
            seed=3,
        )
        c.stroke(
            [(120 + dx, 60 + dy)],
            P.CADMIUM_YELLOW,
            amount=20,
            brush=brushes.ROUND_HOG,
            seed=4,
        )
        knife = Brush(brushes.PALETTE_KNIFE).load(P.TITANIUM_WHITE, 4)
        c.stroke([(70 + dx, 75 + dy), (210 + dx, 60 + dy)], brush=knife, seed=5)
        return c

    a = paint(Canvas(280, 120, texture=0, seed=1), 0, 0)
    b = paint(Canvas(400, 260, texture=0, seed=1), 50, 70)
    for name in ("top", "und", "top_a", "und_a"):
        sa, sb = getattr(a, name), getattr(b, name)[70 : 70 + 120, 50 : 50 + 280]
        # (a knife's flow can leave a speck of 1e-29 differently)
        np.testing.assert_allclose(sa, sb, rtol=0, atol=1e-20, err_msg=name)


OFF = 120  # how far the larger canvas reaches past the smaller one


def _left_after(bt, path, size, width=None, amount=6):
    """The share of its paint a brush has left after ``path`` on a canvas of
    ``size``, and after the same path on a canvas reaching OFF px further
    all round."""
    w = width or bt.width
    out = []
    for off in (0, OFF):
        c = Canvas(size[0] + 2 * off, size[1] + 2 * off, texture=0, seed=0)
        b = Brush(bt)
        b.width = float(w)
        b.load(P.CADMIUM_RED, amount)
        v0 = b.volume
        c.stroke([(x + off, y + off) for x, y in path], brush=b, width=w, seed=7)
        out.append(b.volume / v0)
    return out


@pytest.mark.parametrize(
    ("name", "start"),
    [("FLAT_HOG", -0.3), ("FLAT_HOG", -1.5), ("FLAT_HOG", -4.0), ("ROUND", -4.0)],
)
def test_a_stroke_entering_from_off_the_canvas_arrives_with_its_paint(name, start):
    """A stroke started off the canvas (``start`` widths past the left edge)
    has used up on its way in what it would have used on a larger canvas
    (it arrived with 1.08 to 1.4 times that; a round from far off laid
    nothing at all)."""
    bt = getattr(brushes, name)
    h = int(2 * bt.width) + 20
    cut, big = _left_after(bt, [(start * bt.width, h / 2), (12.0, h / 2)], (200, h))
    assert abs(cut - big) <= 0.02


@pytest.mark.parametrize("n", [2, 8, 20])
def test_a_stroke_across_a_tiny_canvas_uses_its_paint_as_on_a_large_one(n):
    """A hog crossing a canvas a few pixels wide keeps what it would keep
    on a large canvas (it used to keep three times that)."""
    cut, big = _left_after(
        brushes.FLAT_HOG, [(-60.0, n / 2), (n + 60.0, n / 2)], (n, n), width=24
    )
    assert abs(cut - big) <= 0.01


def test_a_dab_at_the_edge_lays_its_visible_part_as_a_whole_dab():
    """A loaded dab cut by the right edge lays on the canvas what the same
    part of a whole dab gets (lift off: see the lift tests)."""
    bt = brushes.ROUND_HOG
    w = bt.width
    size = int(2 * w) + 10
    for share in (0.3, 0.7):
        cx = size - w / 2 + share * w
        vols = []
        for off in (0, OFF):
            c = Canvas(size + 2 * off, size + 2 * off, texture=0, seed=0)
            b = Brush(bt).load(P.CADMIUM_RED, 30)
            c.stroke(
                [(cx + off, size / 2 + off)], brush=b, seed=7, pressure=0.7, lift=0
            )
            vols.append(
                float(c.wet_thickness[off : off + size, off : off + size].sum())
            )
        assert abs(vols[0] / vols[1] - 1) <= 0.05


def test_a_ridge_past_the_edge_goes_there():
    """An overloaded flat pressed along the bottom edge, half of it past it:
    the ridge it squeezes out on that side lies past the edge (it used to go
    back onto the brush, or pile up on the canvas side)."""
    bt = brushes.FLAT_HOG
    w = bt.width
    h = 100
    y = h - w / 2 + 0.5 * w
    vols, left = [], []
    for off in (0, OFF):
        c = Canvas(300 + 2 * off, h + 2 * off, texture=0, seed=0)
        b = Brush(bt).load(P.CADMIUM_RED, 60)
        c.stroke(
            [(30 + off, y + off), (270 + off, y + off)], brush=b, pressure=0.9, seed=3
        )
        vols.append(float(c.wet_thickness[off : off + h, off : off + 300].sum()))
        left.append(b.volume)
    assert abs(vols[0] / vols[1] - 1) <= 0.03
    assert abs(left[0] / left[1] - 1) <= 0.03


def _knife_pair(bt, path, size):
    """Paint laid on the canvas and left on the blade by a knife stroke on a
    canvas of ``size`` and on one reaching OFF px further all round."""
    out = []
    for off in (0, OFF):
        c = Canvas(size[0] + 2 * off, size[1] + 2 * off, texture=0, seed=0)
        b = Brush(bt).load(P.CADMIUM_RED, 6)
        c.stroke([(x + off, y + off) for x, y in path], brush=b, pressure=0.4, seed=3)
        t = c.wet_thickness[off : off + size[1], off : off + size[0]]
        out.append((float(t.astype(np.float64).sum()), b.volume))
    return out


@pytest.mark.parametrize("name", ["PALETTE_KNIFE", "KNIFE_TROWEL"])
@pytest.mark.parametrize("share", [0.5, 0.9])
def test_a_knife_cut_by_the_edge_lays_its_visible_part_as_uncut(name, share):
    """A knife slab along the bottom edge with ``share`` of the blade past
    it lays on the canvas what the uncut slab lays there, and keeps what it
    keeps (the paint under the part past the edge stayed on the blade and
    went down on the visible part: up to 1.8 times as much, and a trowel
    90% off kept 14,000 units of paint against 17)."""
    bt = getattr(brushes, name)
    h = 140
    y = h - bt.width / 2 + share * bt.width
    (vc, bc), (vb, bb) = _knife_pair(bt, [(30.0, y), (370.0, y)], (400, h))
    assert abs(vc / vb - 1) <= 0.06
    assert bc <= bb + 1.0


def test_a_knife_entering_from_off_the_canvas_arrives_with_its_paint():
    """A trowel started a width past the left edge lays on the canvas about
    what the same stroke lays there on a larger canvas (it laid half as
    much: off the canvas it filled the same gap again at every step)."""
    bt = brushes.KNIFE_TROWEL
    (vc, bc), (vb, bb) = _knife_pair(
        bt, [(-0.5 * bt.width, 60.0), (150.0, 60.0)], (300, 120)
    )
    assert abs(vc / vb - 1) <= 0.06
    assert abs(bc - bb) <= 0.05 * max(bb, 1.0)


# ----------------------------------------------------------------------------
# pigment kept by dabs of a multi-colour brush


def _totals(c, brush):
    on_canvas = c.top.sum(axis=(0, 1), dtype=np.float64)
    on_canvas += c.und.sum(axis=(0, 1), dtype=np.float64)
    return on_canvas + (brush.tip + brush.reservoir + brush.heel).sum(axis=0)


def _two_colour(bt):
    return (
        Brush(bt)
        .load(P.CADMIUM_RED, 6, across="center")
        .load(P.TITANIUM_WHITE, 6, across="edges")
    )


def _marbled(bt):
    mix = [(P.CADMIUM_RED, 1), (P.ULTRAMARINE_BLUE, 1), (P.CADMIUM_YELLOW, 1)]
    return Brush(bt).load_mix(mix, 8, mixing=0.2, seed=4)


@pytest.mark.parametrize("name", ["FLAT_HOG", "ROUND_HOG", "STIPPLER", "MOP"])
@pytest.mark.parametrize("load", [_two_colour, _marbled])
@pytest.mark.parametrize("wet", [False, True])
def test_dabs_of_a_multi_colour_brush_keep_every_pigment(name, load, wet):
    """Every band of K, S, volume and body: what is on the canvas and on the
    brush after a dab, and after a stipple of several, is what there was
    before (they were off by up to a few percent, up to 40% on a mop).
    (With the brush's own lift: the peak it pulls up as it leaves is tested
    with heavier loads in test_lift.)"""
    bt = getattr(brushes, name)
    for taps in (1, 6):
        c = Canvas(200, 140, seed=3)
        if wet:
            c.stroke(
                [(30, 70), (170, 70)],
                P.TITANIUM_WHITE,
                amount=8,
                brush=brushes.FLAT_HOG.sized(60),
                seed=2,
            )
        brush = load(bt)
        before = _totals(c, brush)
        if taps == 1:
            c.stroke([(100, 70)], brush=brush, pressure=0.7, seed=1)
        else:
            pts = [(70 + 12 * i, 60 + 9 * (i % 3)) for i in range(taps)]
            press = [0.2, 0.5, 0.8, 0.4, 0.9, 0.3]
            c.stipple(pts, brush=brush, pressure=press)
        after = _totals(c, brush)
        # (the worst found over 9 presets, both loads, wet and dry, 1 and 6
        # taps: 4.4e-9; another configuration has given 2.6e-8)
        np.testing.assert_allclose(after, before, rtol=3e-7, atol=0)


def test_sharing_paint_by_reach_moves_it_with_what_is_in_it():
    """share_by_reach gives each column the volume asked of it and keeps
    every pigment; where all columns hold the same paint it scales each
    column, as it always did."""
    from oilpaint import _engine as eng

    u = np.linspace(-1, 1, 15)[None, :].repeat(9, axis=0)
    taps = eng.bristle_taps(u, 0.2, 2, True, 0.3)
    rng = np.random.default_rng(0)
    d = rng.uniform(0.2, 1.0, u.shape)
    demand = eng.column_demand(taps, d)

    one = np.zeros((eng.COLUMNS, eng.PACKED))
    paint = rng.uniform(0.1, 5.0, eng.PACKED)
    paint[IV] = 1.0
    vols = rng.uniform(0.5, 3.0, eng.COLUMNS)
    one[:] = vols[:, None] * paint[None, :]
    many = one * rng.uniform(0.5, 2.0, one.shape)
    many[:, IV] = vols
    held, asked = vols.sum(), demand.sum()
    for tip in (one.copy(), many):
        before = tip.copy()
        eng.share_by_reach(tip, taps, d, np.empty(0), True)
        np.testing.assert_allclose(tip[:, IV], held * demand / asked, rtol=1e-12)
        np.testing.assert_allclose(tip.sum(axis=0), before.sum(axis=0), rtol=1e-13)
        assert tip.min() >= 0
    # one paint: exactly the old scaling (sums in the engine's order)
    held, asked = 0.0, 0.0
    for b in range(eng.COLUMNS):
        held += one[b, IV]
        asked += demand[b]
    old = one.copy()
    for b in range(eng.COLUMNS):
        old[b] *= held * demand[b] / asked / one[b, IV]
    eng.share_by_reach(one, taps, d, np.empty(0), True)
    assert np.array_equal(old, one)


# ----------------------------------------------------------------------------
# coverage that doesn't hang on a speck of dust


def _dusty_canvas():
    c = Canvas(200, 150, seed=4)
    c.stroke(
        [(20, 40), (180, 50)],
        P.ULTRAMARINE_BLUE,
        amount=10,
        brush=brushes.FLAT_HOG,
        seed=1,
    )
    knife = Brush(brushes.KNIFE_TROWEL).load(P.TITANIUM_WHITE, 6)
    c.stroke([(30, 95), (170, 105)], brush=knife, pressure=0.35, seed=2)
    c.stroke(
        [(50, 10), (70, 140)], P.CADMIUM_RED, amount=6, brush=brushes.ROUND, seed=3
    )
    return c


def _with_dust(c, eps):
    """Nudge the volume of every near-empty layer of a pixel by ``eps``."""
    c = c.copy()
    for layer in (c.top, c.und):
        v = layer[..., IV]
        v[np.abs(v) < 1e-7] += np.float32(eps)
    return c


def _u8(c):
    return np.round(c.render(lighting=False) * 255).astype(np.int16)


def _covered(c, layer):
    """How much of each pixel a layer's paint covers as the renderer sees it:
    its coverage, or less where the paint is too thin to make a film."""
    v = np.maximum(getattr(c, layer)[..., IV], 0.0)
    return np.minimum(getattr(c, layer + "_a"), v / MIN_FILM)


@pytest.mark.parametrize("eps", [1e-8, -1e-8])
def test_specks_of_dust_decide_no_coverage(eps):
    """Volumes of near-empty pixels nudged by 1e-8 either way: a brush
    stroke over the canvas paints the same picture (to a level of the 8-bit
    render), and wherever the paint ends up the same, so does its coverage
    (before, coverage flipped between 1 and 0.3 on such pixels, and the
    render by up to 48 levels)."""
    base = _dusty_canvas()
    runs = []
    for c in (base.copy(), _with_dust(base, eps)):
        c.stroke(
            [(15, 75), (185, 70)],
            P.TITANIUM_WHITE,
            amount=8,
            brush=brushes.FLAT_HOG,
            seed=5,
        )
        runs.append(c)
    a, b = runs
    assert np.abs(_u8(a) - _u8(b)).max() <= 1
    same = np.ones(a.top_a.shape, bool)
    for layer in ("top", "und"):
        same &= (
            np.abs(getattr(a, layer).astype(float) - getattr(b, layer)).max(axis=2)
            < 1e-6
        )
    for layer in ("top", "und"):
        assert np.abs(_covered(a, layer) - _covered(b, layer))[same].max() <= 1e-3


@pytest.mark.parametrize("eps", [1e-8, -1e-8])
def test_specks_of_dust_decide_no_coverage_under_a_knife(eps):
    """The same for a knife dragged through the paint: where the paint ends
    up the same, so does what of the pixel it covers, and its colour to a
    level. (Elsewhere a knife's flow can amplify a speck
    into a different stroke at a pixel or two: that is the flow, not
    coverage, and nothing here holds the render to it.) A guard: at -1e-8
    this passed before the fix too."""
    base = _dusty_canvas()
    runs = []
    for c in (base.copy(), _with_dust(base, eps)):
        knife = Brush(brushes.PALETTE_KNIFE).load(P.CADMIUM_RED, 3)
        # (inside the canvas: past its edge the blade reads the paint at
        # the edge for all it passes over there, and a speck at the edge
        # moves what it lays by as much as a trace, 0.002 thick, needs to
        # take a different coverage)
        c.stroke([(40, 120), (185, 90)], brush=knife, pressure=0.5, seed=7)
        runs.append(c)
    a, b = runs
    same = np.ones(a.top_a.shape, bool)
    for layer in ("top", "und"):
        same &= (
            np.abs(getattr(a, layer).astype(float) - getattr(b, layer)).max(axis=2)
            < 1e-6
        )
    for layer in ("top", "und"):
        assert np.abs(_covered(a, layer) - _covered(b, layer))[same].max() <= 1e-3
    assert np.abs(_u8(a) - _u8(b)).max(axis=2)[same].max() <= 1


def test_no_paint_is_left_below_nothing():
    """Dragging and the knife's flow leave no channel of any layer below zero,
    and a layer with no paint in it covers nothing."""
    c = _dusty_canvas()
    c.stroke([(15, 75), (185, 70)], None, brush=brushes.ROUND_HOG, pressure=0.9, seed=6)
    c.stroke(
        [(100, 5), (120, 145)],
        brush=Brush(brushes.KNIFE_STRAIGHT),
        pressure=0.85,
        seed=8,
    )
    for layer, cov in ((c.top, c.top_a), (c.und, c.und_a)):
        assert layer.min() >= 0
        assert not (cov[layer[..., IV] <= 1e-6] > 0).any()


# ----------------------------------------------------------------------------
# the slanted edge of thick paint


def _slanted(angle, amount):
    s = 260
    c = Canvas(s, s, seed=2)
    t = np.radians(angle)
    tx, ty = np.cos(t), np.sin(t)
    path = [(s / 2 - 100 * tx, s / 2 - 100 * ty), (s / 2 + 100 * tx, s / 2 + 100 * ty)]
    c.stroke(
        path, P.CADMIUM_RED, amount=amount, brush=brushes.FLAT_HOG.sized(36), seed=1
    )
    return c, tx, ty


def _bilinear(a, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, a.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, a.shape[0] - 2)
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    top = a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx
    return top * (1 - fy) + (a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx) * fy


@pytest.mark.parametrize("amount", [30, 120])
def test_paint_outside_a_slanted_edge_covers_no_more_than_the_films_beside_it(amount):
    """Paint pushed one pixel out past the staircase of a slanted edge had no
    coverage, so it rendered as solid as the paint inside: a tooth at every
    step. Now no pixel near paint holds a visible amount of it uncovered."""
    c, _, _ = _slanted(30, amount)
    v = c.top[..., IV]
    covered_near = np.zeros(v.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            covered_near |= np.roll(np.roll(c.top_a >= 0.5, dy, 0), dx, 1)
    assert not ((c.top_a == 0) & (v > 0.05) & covered_near).any()


def test_a_slanted_edge_of_thick_paint_is_a_smooth_ramp():
    """Counting the edge pixels at least 16 levels (8-bit, unlit, luma)
    darker than both their neighbours one pixel along the stroke: this 30
    degree edge of a loaded flat had 191 (a level edge has none). An ideal
    anti-aliased straight edge at 30 degrees, rendered the same way, scores
    about 110 to 150 (the pixel staircase itself, in sRGB); this one now 109."""
    c, tx, ty = _slanted(30, 30)
    lum = 255.0 * (c.render(lighting=False) @ np.array([0.2126, 0.7152, 0.0722]))
    dark = lum[2, 2] - lum
    s = dark.shape[0]
    yy, xx = np.mgrid[0:s, 0:s].astype(float) + 0.5
    along = (xx - s / 2) * tx + (yy - s / 2) * ty
    lat = -(xx - s / 2) * ty + (yy - s / 2) * tx
    a = _bilinear(dark, xx - 0.5 + tx, yy - 0.5 + ty)
    b = _bilinear(dark, xx - 0.5 - tx, yy - 0.5 - ty)
    band = (np.abs(along) < 60) & (np.abs(np.abs(lat) - 18) < 6)
    teeth = int(((dark - np.maximum(a, b) >= 16) & band).sum())
    assert teeth <= 150, teeth


def test_the_level_edge_of_heavy_paint_stays_solid():
    """Along a level edge, paint slumped past what the brush covered lies in
    a band; it is as solid as the paint it came from, all along the edge
    (paint made thin by the volume alone came out pale in patches wherever
    a furrow left a little less of it)."""
    c, _, _ = _slanted(0, 120)
    lum = 255.0 * (c.render(lighting=False) @ np.array([0.2126, 0.7152, 0.0722]))
    dark = lum[2, 2] - lum
    full = dark[130, 80:180].mean()  # inside the stroke
    v = c.top[..., IV]
    edge = np.zeros(v.shape, bool)
    edge[:, 80:180] = True
    # the outer rows: paint with an empty pixel above or below it
    empty = v < 1e-3
    outer = (np.roll(empty, 1, 0) | np.roll(empty, -1, 0)) & ~empty & edge
    thick = outer & (v > 1.0)
    assert thick.sum() > 100
    assert dark[thick].min() >= full - 4


def test_on_a_round_paint_moves_between_bristles_the_short_way_round():
    """A round's columns go round its bundle: the last is next to the first.
    Paint asked of the first column, held over by the last, moves straight
    across that seam; the columns in between keep their own paint."""
    from oilpaint import _engine as eng

    n = eng.COLUMNS
    # one pixel per column, under that column alone, asking for what it
    # holds, but the first asks for 5 more and the last for 5 less
    cols = np.arange(n, dtype=np.int32)[:, None].repeat(n, axis=1)
    wts = np.zeros((n, n))
    wts[:, 0] = 1.0
    ntap = np.ones(n, np.int32)
    rng = np.random.default_rng(1)
    tip = rng.uniform(0.1, 3.0, (n, eng.PACKED))
    tip[:, IV] = 2.0
    tip[n - 1, IV] = 7.0
    d = np.full((1, n), 2.0)
    d[0, 0] = 7.0
    before = tip.copy()
    eng.share_by_reach(tip, eng.Taps(cols, wts, ntap), d, np.empty(0), True)
    np.testing.assert_allclose(tip.sum(axis=0), before.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(tip[:, IV], d[0], rtol=1e-12)
    # the columns between are untouched; the first now holds the last one's
    # paint as well as its own
    np.testing.assert_array_equal(tip[1 : n - 1], before[1 : n - 1])
    moved = before[n - 1] * (5.0 / 7.0)
    np.testing.assert_allclose(tip[0], before[0] + moved, rtol=1e-12)


def test_a_knife_lifted_off_pulls_up_only_its_own_paint():
    """Lifting a knife off pulls its own paint up into crests; the paint of
    earlier strokes under it stays as it lies (scaling it as well made and
    destroyed pigment of colours the knife never carried)."""
    from oilpaint._knife import _lift_off

    c = Canvas(140, 100, seed=3)
    c.stroke(
        [(10, 35), (130, 35)],
        P.CADMIUM_YELLOW,
        amount=40,
        brush=brushes.FLAT_HOG.sized(40),
        seed=1,
    )
    c.stroke(
        [(10, 70), (130, 70)],
        P.PHTHALO_BLUE,
        amount=40,
        brush=brushes.FLAT_HOG.sized(40),
        seed=2,
    )
    before = c.top.copy()
    second = int(c.top_id[70, 70])
    sl = (slice(20, 90), slice(20, 120))
    under = np.ones((70, 100))
    _lift_off(c, sl, under, 1.5, 4.0, 0.0, np.random.default_rng(1), second)
    mine = c.top_id == second
    assert np.array_equal(c.top[~mine], before[~mine])
    assert not np.array_equal(c.top[mine], before[mine])
    v0 = before[mine][:, IV].astype(np.float64).sum()
    assert abs(c.top[mine][:, IV].astype(np.float64).sum() / v0 - 1) < 1e-6


def test_on_a_round_the_paint_shared_by_reach_meets_every_column_s_share():
    """Random rings of bristles, some empty, some full: every column ends with
    the volume asked of it, and every pigment is kept."""
    from oilpaint import _engine as eng

    n = eng.COLUMNS
    rng = np.random.default_rng(7)
    cols = np.arange(n, dtype=np.int32)[:, None].repeat(n, axis=1)
    wts = np.zeros((n, n))
    wts[:, 0] = 1.0
    ntap = np.ones(n, np.int32)
    for _ in range(400):
        tip = rng.uniform(0.1, 3.0, (n, eng.PACKED))
        vols = rng.uniform(0.0, 3.0, n)
        vols[rng.random(n) < 0.3] = 0.0
        if vols.sum() == 0:
            continue
        tip[:, IV] = vols
        tip[vols == 0] = 0.0
        d = rng.uniform(0.0, 3.0, (1, n))
        d[0, rng.random(n) < 0.2] = 0.0
        if d[0, vols > 0].sum() == 0:
            continue
        before = tip.copy()
        eng.share_by_reach(tip, eng.Taps(cols, wts, ntap), d, np.empty(0), True)
        held = vols.sum()
        want = np.where(vols > 0, held * d[0] / d[0, vols > 0].sum(), 0.0)
        np.testing.assert_allclose(tip[:, IV], want, rtol=0, atol=1e-12 * held)
        np.testing.assert_allclose(tip.sum(axis=0), before.sum(axis=0), rtol=1e-12)


def _pushed_case():
    """Paint arrays for fix_coverage: a film 4 thick, fully covered, ending in
    an anti-aliased row at 0.3 with a band of thick paint pushed past it (2.0,
    no coverage), and a faint covered film with a thick speck beside it."""
    from oilpaint import _engine as eng

    h, w = 30, 60
    top = np.zeros((h, w, eng.PACKED), np.float32)
    und = np.zeros_like(top)
    paint = np.linspace(0.5, 3.0, eng.PACKED).astype(np.float32)
    paint[IV] = 1.0
    top_a = np.zeros((h, w), np.float32)
    top[:10, 5:50] = 4.0 * paint
    top_a[:10, 5:50] = 1.0
    top[10, 5:50] = 1.2 * paint  # the edge row, 0.3 covered
    top_a[10, 5:50] = 0.3
    top[11:14, 5:50] = 2.0 * paint  # pushed out, uncovered
    top[20:24, 5:20] = 0.02 * paint  # a faint film
    top_a[20:24, 5:20] = 0.2
    top[22, 20] = 1.0 * paint  # a thick speck beside it
    return top, und, top_a, np.zeros_like(top_a)


def test_coverage_of_pushed_paint_does_not_depend_on_the_order_it_is_taken_in():
    """fix_coverage on paint and on its mirror image (either way) gives the
    mirrored coverage, exactly."""
    from oilpaint import _engine as eng

    top, und, top_a, und_a = _pushed_case()
    eng.fix_coverage(top, und, top_a, und_a, 0, top.shape[0], 0, top.shape[1])
    for axis in (0, 1):
        t2, u2, a2, b2 = (
            np.ascontiguousarray(np.flip(x, axis)) for x in _pushed_case()
        )
        eng.fix_coverage(t2, u2, a2, b2, 0, t2.shape[0], 0, t2.shape[1])
        assert np.array_equal(np.flip(a2, axis), top_a)


def test_thick_paint_pushed_well_past_an_edge_is_covered_by_its_own_volume():
    """Thick paint pushed out past an anti-aliased edge row continues its
    ramp right beside it, and beyond that covers its pixels as its volume
    does, all along (a band of it took the edge row's 0.3 all along); a
    thick speck beside a faint film keeps its coverage."""
    from oilpaint import _engine as eng

    top, und, top_a, und_a = _pushed_case()
    eng.fix_coverage(top, und, top_a, und_a, 0, top.shape[0], 0, top.shape[1])
    assert np.all(top_a[12:14, 8:47] >= 0.99)
    assert top_a[22, 20] >= 0.9


def _kernel_signatures() -> int:
    """How many signatures the compiled kernels have between them (each new
    one is compiled, or loaded from numba's cache, on its first call)."""
    from oilpaint import _dab, _render, spectral
    from oilpaint import _engine as eng

    return sum(
        len(getattr(f, "signatures", ()))
        for module in (eng, _render, _dab, spectral)
        for f in vars(module).values()
    )


def test_a_long_knife_stroke_off_the_canvas_keeps_little_memory():
    """What a knife has laid past the edge is kept only where it laid it: a
    3,000 px diagonal stroke over a small canvas peaks at about 6 MB. A record
    over the stroke's bounding box (2,100 px square) takes 18 MB in float32
    alone, and the stroke with one peaks at about 27 MB (a float64 one took
    40 MB), growing with the square of the stroke.

    A shorter knife stroke from far past the edge, across a throwaway canvas,
    runs through the same kernels with the same signatures first: compiling
    them inside the measurement (with numba's cache empty, as on CI) took the
    peak to 18 MB and more, whatever the record kept."""
    import tracemalloc

    def knife():
        return Brush(brushes.PALETTE_KNIFE).load(P.CADMIUM_RED, 6)

    Canvas(200, 200, seed=1).stroke(
        [(-400, -400), (100, 100)], brush=knife(), pressure=0.5, seed=1
    )
    c, k = Canvas(200, 200, seed=1), knife()
    compiled = _kernel_signatures()
    tracemalloc.start()
    c.stroke([(-950, -950), (1150, 1150)], brush=k, pressure=0.5, seed=1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert _kernel_signatures() == compiled, "a kernel compiled in the measurement"
    assert peak < 12e6


@pytest.mark.parametrize(
    "x, floor, ceil",
    [
        (3.0, 3, 3),
        (np.nextafter(3.0, 0.0), 3, 3),  # (a whole number, to rounding)
        (np.nextafter(3.0, 9.0), 3, 3),
        (3.0 - 4e-7, 3, 3),  # within the slack
        (3.0 + 4e-7, 3, 3),
        (3.0 - 2e-6, 2, 3),  # outside it
        (3.0 + 2e-6, 3, 4),
        (2.5, 2, 3),  # exact halves are no whole numbers
        (-2.5, -3, -2),
        (-2.0, -2, -2),
        (np.nextafter(-2.0, 0.0), -2, -2),
        (np.nextafter(-2.0, -9.0), -2, -2),
        (-2.0 - 2e-6, -3, -2),
        (-2.0 + 2e-6, -2, -1),
        (0.0, 0, 0),
        (-1e-9, 0, 0),
        (1e-9, 0, 0),
        (1e6 + 0.5, 1000000, 1000001),
    ],
)
def test_counts_from_ratios_take_a_hair_off_a_whole_number_as_that_number(
    x, floor, ceil
):
    from oilpaint._engine import WHOLE_SLACK, whole_ceil, whole_floor

    assert WHOLE_SLACK == 1e-6
    assert whole_floor(float(x)) == floor
    assert whole_ceil(float(x)) == ceil
