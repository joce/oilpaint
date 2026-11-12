"""A lightly pressed brush carrying little paint over dried relief: its tips
ride over the hollows deeper than they reach and leave its paint on the
crests (scumbling); pressed harder, or loaded, it goes into them as before.
Bare canvas and the weave are none of this's business (see ``tooth``)."""

import pickle
from itertools import pairwise

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes, surfaces
from oilpaint import _engine as eng
from oilpaint import palette as P

W, H = 260, 90
HOG = brushes.FLAT_HOG.sized(30)
SEEDS = (1, 2, 3)


def _furrowed(depth=2.0, period=9.0, w=W, h=H, x0=0.0, texture=None):
    """A canvas with dried furrows ``depth`` deep across it, every
    ``period`` px along x (the crests at x = x0, x0 + period, ...)."""
    c = Canvas(w, h, seed=1, texture=texture)
    x = np.arange(w) - x0
    ridge = depth * (0.5 + 0.5 * np.cos(2 * np.pi * x / period))
    c.dry_height[:] = ridge[None, :].astype(np.float32)
    c.dry()
    return c


def _flat(height=1.0):
    """A canvas with a flat dried film ``height`` thick all over."""
    c = Canvas(W, H, seed=1)
    c.dry_height[:] = height
    c.dry()
    return c


def _drag(c, p, amount, seed, path=None, bt=HOG):
    """Drag a brush ``bt`` (a hog flat) loaded with ``amount`` of white along
    ``path`` (by default the length of the canvas, across the furrows); the
    brush after and what it held before."""
    b = Brush(bt).load(P.TITANIUM_WHITE, amount)
    v0 = b.volume
    c.stroke(path or [(20, H / 2), (W - 20, H / 2)], brush=b, pressure=p, seed=seed)
    return b, v0


def _band(c, cols=slice(30, W - 30)):
    """The middle of the stroke: rows inside it, away from its ends (or the
    columns ``cols``)."""
    m = np.zeros(c.dry_height.shape, bool)
    m[H // 2 - 8 : H // 2 + 8, cols] = True
    return m


def _laid(c, base, cols=slice(30, W - 30)):
    """What the stroke laid in its middle (volume per pixel), and the dried
    relief under it (height above its mean)."""
    m = _band(c, cols)
    laid = (c.top_v + c.und_v).astype(np.float64)[m]
    rel = (base.dry_height - base.dry_height.mean()).astype(np.float64)[m]
    return laid, rel


def _on_crests(c, base):
    """The share of what the stroke laid in its middle that lies on the upper
    half of the relief."""
    laid, rel = _laid(c, base)
    return laid[rel > 0].sum() / laid.sum()


def _hollows_covered(c, base, cols=slice(30, W - 30)):
    """The share of the deep hollows (more than 0.6 below the mean) in the
    middle of the stroke (or in columns ``cols``) that it covered."""
    laid, rel = _laid(c, base, cols)
    return float((laid[rel < -0.6] > 0.15).mean())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", [0.15, 0.2])
def test_a_light_dry_stroke_lays_its_paint_on_the_crests(seed, p):
    """A nearly dry hog dragged lightly across dried furrows leaves its paint
    on the crests and the hollows bare (it laid a solid patch over both,
    uncorrelated with the relief: r about 0, half on the upper half)."""
    base = _furrowed()
    c = base.copy()
    _drag(c, p, 1.5, seed)
    laid, rel = _laid(c, base)
    assert np.corrcoef(laid, rel)[0, 1] > 0.35
    assert _on_crests(c, base) > 0.85
    assert _hollows_covered(c, base) < 0.05


def test_it_skips_less_as_it_is_pressed_harder_or_carries_more():
    """The share of its paint on the crests falls as the brush is pressed
    harder (p 0.2, 0.4, 0.7 with a little paint) and as it carries more
    (amount 1.5, 3, 8 at p 0.2), from almost all of it to about half."""
    base = _furrowed()
    for seed in SEEDS:
        by_p, by_load = [], []
        for p in (0.2, 0.4, 0.7):
            c = base.copy()
            _drag(c, p, 3.0, seed)
            by_p.append(_on_crests(c, base))
        for amount in (1.5, 3.0, 8.0):
            c = base.copy()
            _drag(c, 0.2, amount, seed)
            by_load.append(_on_crests(c, base))
        for shares in (by_p, by_load):
            assert shares[0] > 0.8 and shares[-1] < 0.6
            assert all(a > b + 0.05 for a, b in pairwise(shares))


def _totals(c, b):
    canvas = c.top.sum(axis=(0, 1), dtype=np.float64) + c.und.sum(
        axis=(0, 1), dtype=np.float64
    )
    return canvas + (b.tip + b.reservoir + b.heel).sum(axis=0)


@pytest.mark.parametrize("amount", [1.5, 3.0])
def test_the_paint_it_skips_stays_on_the_brush(amount):
    """Over the furrows a light stroke keeps what it does not lay in the
    hollows (and goes on with it): after the same short path it has far
    more left than over a flat dried film (it had as much left, having laid
    the hollows' share on the crests). Every channel is conserved."""
    path = [(20, H / 2), (110, H / 2)]
    for seed in SEEDS:
        left = []
        for base in (_furrowed(), _flat()):
            c = base.copy()
            b = Brush(HOG).load(P.TITANIUM_WHITE, amount)
            v0 = b.volume
            before = _totals(c, b)
            c.stroke(path, brush=b, pressure=0.2, seed=seed)
            after = _totals(c, b)
            np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-9)
            left.append(b.volume / v0)
        assert left[0] > 1.4 * left[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_loaded_stroke_at_ordinary_pressure_still_covers_the_hollows(seed):
    """In the same furrows that a light, nearly dry stroke leaves bare, a
    loaded one pressed as usual goes on covering the hollows: the paint
    standing on its tips, and the pressure, take them down into them."""
    base = _furrowed()
    c = base.copy()
    _drag(c, 0.2, 1.5, seed)
    assert _hollows_covered(c, base) < 0.05
    for p, amount in ((0.5, 12.0), (0.7, 8.0), (0.9, 12.0)):
        c = base.copy()
        _drag(c, p, amount, seed)
        assert _hollows_covered(c, base) > 0.85, (p, amount)


def test_pressure_alone_takes_the_tips_down_into_the_hollows():
    """With the same paint on the brush, pressing harder takes the tips down
    into hollows they rode over: none of them covered at p 0.3, almost all
    at p 0.5 (on the first stretch, before the brush runs low)."""
    base = _furrowed()
    first = slice(30, 100)
    for seed in SEEDS:
        covered = []
        for p in (0.3, 0.5):
            c = base.copy()
            _drag(c, p, 3.0, seed)
            covered.append(_hollows_covered(c, base, first))
        assert covered[0] < 0.05 and covered[1] > 0.85


def test_soft_hair_reaches_further_into_the_hollows_than_stiff():
    """Two hog flats differing only in stiffness, pressed alike: the soft one
    bends down into the hollows the stiff one rides over."""
    base = _furrowed()
    first = slice(30, 100)
    for seed in SEEDS:
        covered = []
        for stiffness in (0.3, 0.85):
            c = base.copy()
            _drag(c, 0.4, 3.0, seed, bt=HOG.with_(stiffness=stiffness))
            covered.append(_hollows_covered(c, base, first))
        assert covered[0] > 0.8 and covered[1] < 0.5


def test_what_it_rides_over_does_not_count_as_running_dry():
    """A light stroke across a field of tall, sparse dried points rides over
    nearly all of it (the ground between them is far below the mean around
    it) and comes out the other side with its paint, to lay it on the bare
    canvas beyond: what is out of reach does not count as the brush having
    run dry (it laid next to nothing beyond, as on main)."""
    c = Canvas(W, H, seed=1, texture=0)
    yy, xx = np.mgrid[0:H, 0:W]
    field = (xx >= 40) & (xx < 140) & (xx % 9 == 0) & (yy % 9 == 0)
    c.dry_height[:] = np.where(field, 50.0, 0.0)
    c.dry()
    for seed in SEEDS:
        d = c.copy()
        _drag(d, 0.2, 3.0, seed)
        assert (d.top_v + d.und_v)[:, 160:].sum() > 300.0


def _forced(c):
    """``c``, made to look for what is out of reach at every pixel (as if its
    surface had hollows far deeper than the weave's)."""
    c._beyond_weave = lambda: 10.0
    return c


@pytest.mark.parametrize("name", ["CANVAS", "LINEN_COARSE", "GESSO_PANEL", "OIL_PAPER"])
def test_bare_canvas_and_a_flat_film_are_within_reach(name):
    """However lightly it is pressed, a brush reaches everywhere on bare
    canvas (the hollows of the surface are the weave's, to the bit, whatever
    the weave) and on a flat dried film over it: it need not look (see
    Canvas._beyond_weave), and looking at every pixel changes nothing. (A
    film following the weave, thicker in its hollows, used to send a light
    stroke onto the weave's hollows.)"""
    surf = getattr(surfaces, name)
    bare = Canvas(120, 70, seed=4, surface=surf)
    np.testing.assert_array_equal(bare._weave_hollows(), bare._relief().hollows)
    film = bare.copy()
    film.dry_height[:] = 0.8
    film.dry()
    for c in (bare, film):
        assert c._beyond_weave() <= eng.REACH_MIN
        a, b = c.copy(), _forced(c.copy())
        for d in (a, b):
            _drag(d, 0.15, 1.5, 3, path=[(10, 35), (110, 35)])
        for attr in ("top", "und", "top_a", "und_a"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr), attr)


def _same_paint(a, b):
    for attr in ("top", "und", "top_a", "und_a", "top_id"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr), attr)


def test_what_a_canvas_works_out_of_its_relief_is_never_stale():
    """What a canvas keeps worked out of its relief is made again when its
    paint dries, and not pickled or copied: a stroke over relief dried after
    a first stroke scumbles as on a twin that never kept it; a pickle holds
    nothing more than it did; a pickled or copied canvas with dried relief
    paints on byte for byte as the original; and one pickled without the
    weave's hollows makes them again as a new canvas has them."""
    c = Canvas(W, H, seed=1)
    _drag(c, 0.5, 6.0, 1, path=[(20, 20), (120, 20)])  # (on bare canvas)
    twin = pickle.loads(pickle.dumps(c))
    x = np.arange(W)
    for d in (c, twin):
        d.dry_height[:] = 2.0 * (0.5 + 0.5 * np.cos(2 * np.pi * x / 9.0))
        d.dry()
        _drag(d, 0.2, 1.5, 2)
    _same_paint(c, twin)
    state = c.__getstate__()
    assert "_beyond_weave_at" not in state and "_weave_hollows_map" not in state
    b, e = pickle.loads(pickle.dumps(c)), c.copy()
    for d in (c, b, e):
        _drag(d, 0.2, 1.5, 3)
    _same_paint(c, b)
    _same_paint(c, e)
    new = Canvas(W, H, seed=1)
    np.testing.assert_array_equal(b._weave_hollows(), new._relief().hollows)


M = 60  # how far the larger canvas reaches past the smaller one


@pytest.mark.parametrize("share", [0.3, 0.6])
def test_a_scumble_cut_by_the_edge_skips_as_on_a_larger_canvas(share):
    """A light stroke along the bottom edge, ``share`` of its width off the
    canvas, over furrows running down the canvas, uses up on its way what
    the same stroke uses on a larger canvas, and lays what it lays there:
    off the canvas its tips ride over the hollows as on it."""
    w = HOG.width
    y = H - w / 2 + share * w
    path = [(20.0, y), (W - 20.0, y)]
    out = []
    for off in (0, M):
        c = _furrowed(w=W + 2 * off, h=H + 2 * off, x0=off, texture=0)
        base = c.copy()
        b = Brush(HOG).load(P.TITANIUM_WHITE, 3.0)
        v0 = b.volume
        c.stroke([(x + off, y + off) for x, y in path], brush=b, pressure=0.2, seed=5)
        laid = (c.top_v + c.und_v)[off : off + H, off : off + W].astype(np.float64)
        rel = base.dry_height[off : off + H, off : off + W] - 1.0
        out.append((b.volume / v0, laid, rel))
    (cut, t_cut, rel), (big, t_big, _) = out
    assert abs(cut - big) <= 0.02
    rows = slice(int(y - w / 2) + 2, H - 2)
    cols = slice(30, W - 30)
    a, b = t_cut[rows, cols], t_big[rows, cols]
    assert abs(a.sum() / b.sum() - 1) <= 0.05
    # (and it does skip the hollows)
    r = rel[rows, cols]
    assert b[r > 0].sum() / b.sum() > 0.8
