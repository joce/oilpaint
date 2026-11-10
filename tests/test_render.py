"""Rendering: the colour of the paint, and the light on its relief."""

from typing import Any

import numpy as np
import pytest

from oilpaint import Canvas, Paint, brushes, surfaces
from oilpaint import palette as P
from oilpaint import spectral as sp
from oilpaint.canvas import _composite

RAKING = (-0.6, -0.8, 0.7)


def _layer(c, paint, volume, wet=True):
    """Lay ``paint`` ``volume`` thick (an (H, W) map, or a number) straight
    into the canvas: as a wet top layer, or dried."""
    volume = np.broadcast_to(np.asarray(volume, np.float32), (c.height, c.width))
    if wet:
        N = sp.N_BANDS
        c.top[..., :N] = paint.K * volume[..., None]
        c.top[..., N] = paint.S * volume
        c.top[..., N + 1] = volume
        c.top[..., N + 2] = paint.body * volume
        c.top_a[:] = 1.0
    else:
        c.dry_height[:] = volume
        c._update_tooth()
    return c


def _ridge(width=120, height=90, top=12.0, sigma=6.0, wet=True, paint="#3060a0"):
    """A straight ridge of thick paint running down the canvas (along y), on
    a smooth support, with a little paint everywhere around it."""
    c = Canvas(width, height, surface=surfaces.GLASS.with_(transparent=False))
    x = np.arange(width, dtype=np.float32)
    profile = 2.0 + top * np.exp(-0.5 * ((x - width / 2) / sigma) ** 2)
    return _layer(c, Paint(paint), np.tile(profile, (height, 1)), wet)


def test_unlit_render_is_the_colour_of_the_paint():
    c = Canvas(90, 60, seed=3)
    c.stroke([(10, 30), (80, 30)], P.CADMIUM_RED, amount=8, width=24, seed=1)
    c.stroke([(45, 5), (45, 55)], P.ULTRAMARINE_BLUE, amount=8, width=20, seed=2)
    img = c.render(lighting=False)
    want = sp.reflectance_to_srgb(c.reflectance())
    assert img.dtype == np.float32 and img.shape == (60, 90, 3)
    assert np.abs(img - want).max() < 1e-4


def test_compiled_colour_matches_the_layer_model():
    c = Canvas(90, 60, seed=3)
    c.stroke([(10, 30), (80, 30)], P.CADMIUM_RED, amount=8, width=24, seed=1)
    c.dry()
    c.stroke([(10, 35), (80, 20)], Paint.mix((P.RED_LAKE, 1), (P.MEDIUM, 3)), seed=4)
    c.stroke([(45, 5), (45, 55)], P.LEAD_WHITE, amount=3, width=16, seed=2)
    R = _composite(c.und_k, c.und_s, c.und_v, c.und_a, c.ground)
    R = _composite(c.top_k, c.top_s, c.top_v, c.top_a, R)
    want = sp.reflectance_to_linear_rgb(R)
    assert np.abs(c._render_linear() - want).max() < 1e-5


def test_flat_paint_lit_from_any_side_keeps_its_colour():
    c = _layer(
        Canvas(80, 60, surface=surfaces.GLASS.with_(transparent=False)),
        P.CADMIUM_ORANGE,
        6.0,
    )
    flat = c.render(lighting=False)
    for light in ((-0.6, -0.8, 1.1), RAKING, (0.9, 0.1, 0.4), (0.0, 0.0, 1.0)):
        lit = c.render(light=light)  # border pixels included
        assert np.abs(lit - flat).max() < 2e-3, light


def test_flat_dry_paint_and_bare_canvas_keep_their_colour_on_average():
    # the weave shades a little from thread to thread, but neither darkens nor
    # lightens the picture as a whole
    for c in (
        Canvas(120, 90, seed=5),
        _layer(Canvas(120, 90, seed=5), P.VIRIDIAN, 4.0, wet=False),
    ):
        lit, flat = c.render(light=RAKING), c.render(lighting=False)
        assert abs(lit.mean() - flat.mean()) < 0.01
        assert np.abs(lit - flat).max() < 0.15


def test_a_ridge_is_lit_on_the_side_facing_the_light_and_shaded_on_the_other():
    c = _ridge()
    flat = c.render(lighting=False).mean(-1)
    for light, lit_side in (
        ((-0.8, 0.0, 0.6), slice(48, 56)),
        ((0.8, 0.0, 0.6), slice(64, 72)),
    ):
        lit = c.render(light=light, gloss=0.0).mean(-1)
        dark_side = slice(64, 72) if lit_side.start < 60 else slice(48, 56)
        row = slice(30, 60)
        assert (lit[row, lit_side] > flat[row, lit_side] + 0.05).all(), light
        assert (lit[row, dark_side] < flat[row, dark_side] - 0.05).all(), light


def test_a_ridge_casts_a_soft_shadow_that_fades_away_from_it():
    c = _ridge(top=20.0, sigma=3.0)
    flat = c.render(lighting=False).mean(-1)
    lit = c.render(light=(-0.9, 0.0, 0.35), gloss=0.0).mean(-1)
    ratio = (lit / flat)[30:60].mean(0)
    beyond = ratio[70:110]  # the paint behind the ridge, away from the light
    assert beyond[0] < 0.6  # in shadow at the foot of the ridge...
    assert beyond[-1] > 0.95  # ...lit again further away
    assert (np.diff(beyond) > -1e-3).all()  # getting lighter all the way: soft


def test_wet_impasto_glints_and_gloss_scales_it():
    c = _ridge()
    light = (-0.8, 0.0, 0.6)
    matte = c.render(light=light, gloss=0.0)
    glossy = c.render(light=light)
    extra = (glossy - matte).mean(-1)[30:60]
    # a highlight on the slope facing halfway between the lamp and the viewer,
    # whiter than the paint (it adds to all three channels)...
    assert extra.max() > 0.2
    assert (glossy - matte)[30:60].min(-1).max() > 0.15
    # ...and none on the far side, nor on the flat paint around it
    assert np.abs(extra[:, 62:]).max() < 1e-3
    assert np.abs(extra[:, :20]).max() < 1e-3
    # dried paint is duller
    dry = _ridge(wet=False)
    extra_dry = (dry.render(light=light) - dry.render(light=light, gloss=0.0)).mean(-1)
    assert extra_dry.max() < extra.max()


def test_highlights_roll_off_rather_than_clip():
    img = _ridge(paint="#f4f2ee").render(light=(-0.8, 0.0, 0.6))
    assert img.max() <= 1.0 and img.min() >= 0.0
    # no flat plateau of clipped white across the highlight
    assert (img >= 1.0 - 1e-6).all(-1).sum() == 0


def test_concurrent_renders_run_and_match_serial_ones():
    # the kernels release the GIL: renders in threads give the same pixels
    import threading

    c = Canvas(160, 100, seed=7)
    c.stroke(
        [(10, 50), (150, 50)],
        P.TITANIUM_WHITE,
        amount=40,
        brush=brushes.FLAT_HOG.sized(30),
        seed=3,
    )
    jobs: list[dict[str, Any]] = [
        {"light": RAKING},
        {"light": (0.3, -0.9, 0.5)},
        {"lighting": False},
    ]
    want = [c.render(**kw) for kw in jobs]
    got: dict[int, np.ndarray] = {}

    def work(i):
        got[i] = c.render(**jobs[i % 3])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(np.array_equal(got[i], want[i % 3]) for i in range(6))


def test_relief_scales_the_light_and_shade():
    c = _ridge()
    flat = c.render(lighting=False)
    low = np.abs(c.render(light=RAKING, relief=0.3, gloss=0.0) - flat).mean()
    high = np.abs(c.render(light=RAKING, relief=1.0, gloss=0.0) - flat).mean()
    none = np.abs(c.render(light=RAKING, relief=0.0, gloss=0.0) - flat).max()
    assert none < 1e-3 and 0 < low < high


def test_reverse_glass_is_not_lit():
    c = Canvas(100, 80, surface=surfaces.GLASS, reverse=True, seed=1)
    c.stroke(
        [(10, 40), (90, 40)],
        P.CADMIUM_YELLOW,
        amount=40,
        brush=brushes.FLAT_HOG.sized(30),
        seed=1,
    )
    assert np.array_equal(c.render(light=RAKING), c.render(lighting=False))


@pytest.mark.parametrize(
    "kw",
    [
        {"light": (1.0, 0.0, 0.0)},
        {"light": (1.0, 0.0, -1.0)},
        {"light": (np.nan, 0.0, 1.0)},
        {"light": (0.0, 0.0, np.inf)},
        {"light": (1.0, 0.0)},
        {"light": (1.0, 0.0, 1.0, 5.0)},
        {"light": "up"},
        {"relief": -1.0},
        {"relief": np.nan},
        {"gloss": -0.5},
        {"gloss": np.inf},
    ],
)
def test_render_rejects_a_bad_light_relief_or_gloss_before_any_work(kw, monkeypatch):
    c = _ridge()

    def no_work():
        raise AssertionError("rendered before checking")

    monkeypatch.setattr(c, "_render_linear", no_work)
    with pytest.raises(ValueError, match=next(iter(kw))):
        c.render(**kw)


def test_unlit_and_reverse_renders_ignore_light_relief_and_gloss():
    c = _ridge()
    flat = c.render(lighting=False)
    options: list[dict[str, Any]] = [
        {"light": None},
        {"light": (1.0, 0.0, -1.0)},
        {"relief": -1.0},
    ]
    for kw in options:
        assert np.array_equal(c.render(lighting=False, **kw), flat)
    g = Canvas(60, 40, surface=surfaces.GLASS, reverse=True, seed=1)
    g.stroke([(5, 20), (55, 20)], P.CADMIUM_YELLOW, amount=20, width=16, seed=1)
    assert np.array_equal(g.render(light=None, gloss=-1.0), g.render(lighting=False))


def test_light_none_is_the_default_lamp():
    from oilpaint.canvas import DEFAULT_LIGHT

    c = _ridge()
    want = c.render()
    assert np.array_equal(c.render(light=None), want)
    assert np.array_equal(c.render(light=DEFAULT_LIGHT), want)
    assert not np.array_equal(c.render(light=RAKING), want)


def test_an_extreme_but_valid_light_is_normalised_safely():
    c = _ridge()
    want = c.render(light=(1.0, 0.0, 1.0))
    for light in ((1e300, 0.0, 1e300), (1e-300, 0.0, 1e-300)):
        assert np.array_equal(c.render(light=light), want), light


def test_tiny_canvases_render():
    for w, h in ((1, 1), (1, 7), (7, 1), (2, 2), (40, 1), (1, 40)):
        c = Canvas(w, h, seed=1)
        c.stroke([(0, 0), (w - 0.99, h - 0.99)], P.CADMIUM_RED, amount=8, width=4)
        options: list[dict[str, Any]] = [{"lighting": False}, {"light": RAKING}, {}]
        for kw in options:
            img = c.render(**kw)
            assert img.shape == (h, w, 3) and np.isfinite(img).all(), (w, h, kw)


def test_gloss_zero_adds_no_highlight_at_all():
    c = _ridge()
    light = (-0.8, 0.0, 0.6)
    lin = c._render_linear()
    L = np.asarray(light) / np.linalg.norm(light)
    no_gloss = c._light(lin, L, 1.0, 0.0)
    # the same shading with every highlight term switched off by hand
    from oilpaint import canvas as cm

    saved = cm.GLOSS_WET, cm.GLOSS_DRY
    cm.GLOSS_WET = cm.GLOSS_DRY = 0.0
    try:
        assert np.array_equal(no_gloss, c._light(lin, L, 1.0, 1.0))
    finally:
        cm.GLOSS_WET, cm.GLOSS_DRY = saved


def test_paint_sunk_into_an_absorbent_ground_shines_less():
    # wet paint on a slope that mirrors the lamp to the viewer
    from oilpaint import _render

    c = _layer(
        Canvas(40, 30, surface=surfaces.GLASS.with_(transparent=False)),
        Paint("#3060a0"),
        4.0,
    )
    L = np.array([-0.8, 0.0, 0.6])
    H = L + [0.0, 0.0, 1.0]
    mx = -H[0] / H[2]
    plane = (mx * np.arange(40, dtype=np.float32))[None, :].repeat(30, 0)
    c.surface_height = lambda relief=1.0: plane  # type: ignore[method-assign]
    lin = np.zeros((30, 40, 3), np.float32)  # black paint: only the highlight

    def highlight(matte):
        v = float(c._light(lin, L, 1.0, 1.0, matte)[15, 20, 0])
        return v / np.sqrt(1 - v * v)  # undo the roll-off towards white

    full = highlight(None)
    half = highlight(np.full((30, 40), 0.5, np.float32))
    assert full > 0.2
    assert half == pytest.approx(full * (1 - 0.5 * _render.MATTE_GLOSS_LOSS), rel=0.02)


def test_compiled_box_blur_matches_the_filter():
    from oilpaint import _render
    from oilpaint._filters import _box_blur

    a = (np.random.default_rng(0).random((37, 53)) * 10).astype(np.float32)
    for r in (0, 1, 3, 20, 60):
        out = np.empty_like(a)
        _render.box_blur(a, r, out)
        assert np.abs(out - _box_blur(a, r)).max() < 1e-5, r


def test_a_long_shadow_fades_out_rather_than_stopping():
    # a very tall ridge under a grazing lamp: its shadow would reach past
    # the end of the search; it fades away smoothly instead of a hard edge
    c = _ridge(width=560, height=40, top=110.0, sigma=4.0, paint="#c8c4bc")
    row = c.render(light=(-0.97, 0.0, 0.25), gloss=0.0).mean(-1)[20, 300:]
    assert row[-1] - row[0] > 0.2  # lit again at the far end
    d = np.diff(row)
    assert (d > -1e-4).all()  # lightening all the way
    # smoothly: a step of a few levels (0.012 per level) would show
    assert np.abs(d).max() < 0.025 and np.abs(np.diff(d)).max() < 0.012


def test_gently_modelled_paint_shows_no_veil_of_gloss():
    # slopes far from the one that mirrors the lamp to the viewer: the oil
    # reflects nothing towards the viewer, wet or dry
    light = (-0.8, 0.0, 0.6)
    for wet in (True, False):
        c = _ridge(top=1.5, sigma=10.0, wet=wet, paint="#2a1a10")
        assert np.array_equal(c.render(light=light), c.render(light=light, gloss=0.0))


def _lab(img):
    return sp.srgb_to_lab(img.reshape(-1, 3).astype(np.float64))


def test_a_thin_dark_painting_keeps_its_colour_and_its_darks_under_the_lamp():
    c = Canvas(160, 120, seed=11, ground="#5a3a22")
    rng = np.random.default_rng(3)
    for k in range(14):
        y = 10 + 8 * k
        paint = (P.BURNT_UMBER, P.BONE_BLACK, P.YELLOW_OCHRE, P.LEAD_WHITE)[k % 4]
        c.stroke(
            [(5, y + rng.normal(0, 3)), (80, y + rng.normal(0, 3)), (155, y)],
            paint,
            amount=6,
            brush=brushes.FLAT_HOG.sized(18),
            pressure=0.5,
            seed=k,
        )
        if k == 6:
            c.dry()
    flat = _lab(c.render(lighting=False))
    for gloss in (0.08, 1.0):
        lit = _lab(c.render(light=RAKING, relief=0.9, gloss=gloss))
        chroma = np.hypot(lit[:, 1], lit[:, 2]).mean()
        assert chroma >= 0.98 * np.hypot(flat[:, 1], flat[:, 2]).mean(), gloss
        dark = flat[:, 0] <= np.percentile(flat[:, 0], 10)
        assert lit[dark, 0].mean() <= flat[dark, 0].mean() + 0.5, gloss
        assert lit[:, 2].mean() >= flat[:, 2].mean() - 0.5, gloss  # not cooler


def test_white_impasto_facing_the_lamp_reaches_near_white():
    c = _ridge(paint="#f0eee8")
    flat = c.render(lighting=False).mean(-1)
    lit = c.render(light=(-0.8, 0.0, 0.6)).mean(-1)
    assert lit[30:60, 40:58].max() > 0.985 and lit[30:60, 40:58].max() > flat.max()


# --------------------------------------------------------------------------
# glints along slanting crests: lines, not beads at the pixel staircase


def _furrowed_ridge(angle, seed=1, furrows=0.3, n=120):
    """A narrow crest of wet paint running at ``angle`` degrees, with fine
    furrows along it (the crest of a loaded brush), on a smooth support, and
    a lamp turned with it: only how it falls on the pixels changes with the
    angle. Returns the canvas, the lamp and the direction of the crest."""
    c = _layer(
        Canvas(n, n, surface=surfaces.GLASS.with_(transparent=False)),
        Paint("#3060a0"),
        4.0,
    )
    a = np.radians(angle)
    tx, ty = np.cos(a), np.sin(a)
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    across = -xx * ty + yy * tx
    along = xx * tx + yy * ty
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.15, 0.45, (64, 1, 1))
    drift = 0.05 * rng.normal(size=(64, 1, 1))
    phase = rng.uniform(0, 2 * np.pi, (64, 1, 1))
    texture = np.cos(2 * np.pi * freq * across + drift * along + phase).sum(0) / 8
    h = (4.0 * np.exp(-0.5 * (across / 1.5) ** 2) + furrows * texture).astype(
        np.float32
    )
    c.surface_height = lambda relief=1.0: h  # type: ignore[method-assign]
    light = np.array([np.sin(a), -np.cos(a), 0.6])
    return c, light / np.linalg.norm(light), (tx, ty)


def _highlight(c, light):
    """The highlight alone, in 8-bit levels, on mid-grey paint."""
    grey = np.full((c.height, c.width, 3), 0.18, np.float32)

    def levels(gloss):
        lit = c._light(grey, light, 1.0, gloss).astype(np.float64)
        return 255 * sp.linear_to_srgb(lit).mean(-1)

    return levels(1.0) - levels(0.0)


def _bilinear(a, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, a.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, a.shape[0] - 2)
    fx, fy = x - x0, y - y0
    top = a[y0, x0] * (1 - fx) + a[y0, x0 + 1] * fx
    return top * (1 - fy) + (a[y0 + 1, x0] * (1 - fx) + a[y0 + 1, x0 + 1] * fx) * fy


def _bead_share(S, direction):
    """Share of the lit pixels (8 levels or more) in the middle of the
    canvas that stand alone along the crest: 8 levels or more above, and
    over twice as bright as, both neighbours one pixel along it. A line,
    even one pixel wide at any slant, has neighbours along it more than
    half as bright; a string of beads does not."""
    n = S.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    tx, ty = direction
    nb = np.maximum(_bilinear(S, xx + tx, yy + ty), _bilinear(S, xx - tx, yy - ty))
    mid = np.zeros(S.shape, bool)
    mid[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = True
    lit = mid & (S >= 8)
    beads = lit & (S - nb >= 8) & (nb < 0.5 * S)
    return beads.sum() / max(lit.sum(), 1)


def test_a_slanting_crest_glints_in_a_line_as_a_level_one_does():
    # point by point, a sharp highlight picks out each step of a slanting
    # crest on the pixels: beads. Gathered along the crest it is a line.
    for seed in (1, 2):
        level = _bead_share(_highlight(*_furrowed_ridge(0, seed)[:2]), (1.0, 0.0))
        for angle in (22, 30, 60):
            c, light, along = _furrowed_ridge(angle, seed)
            slanting = _bead_share(_highlight(c, light), along)
            # (each pixel on its own: 0.25 to 0.4 at these angles)
            assert slanting <= level + 0.05, (seed, angle, level, slanting)


def _glint_width(S):
    """Full width at half maximum (px) across a crest running along the rows,
    of the highlight averaged along it over the middle of the canvas; and
    its peak."""
    n = S.shape[0]
    p = S[:, n // 4 : 3 * n // 4].mean(1)
    i = int(np.argmax(p))
    half = 0.5 * p[i]
    a = i
    while p[a - 1] > half:
        a -= 1
    b = i
    while p[b + 1] > half:
        b += 1
    # interpolated where the profile crosses half its height
    lo = a - (p[a] - half) / (p[a] - p[a - 1])
    hi = b + (p[b] - half) / (p[b] - p[b + 1])
    return hi - lo, p[i]


def test_the_glint_along_a_level_crest_stays_narrow(monkeypatch):
    from oilpaint import canvas as cm

    for furrows, seed in ((0.0, 1), (0.3, 1), (0.3, 2)):
        c, light, _ = _furrowed_ridge(0, seed, furrows)
        width, peak = _glint_width(_highlight(c, light))
        if not furrows:
            assert width < 2.2, width  # a crest 1.5 px across: a fine line
        # as narrow and as bright as each pixel's own highlight, not gathered
        with monkeypatch.context() as m:
            m.setattr(cm, "GLOSS_SPAN", 0.0)
            own_width, own_peak = _glint_width(_highlight(c, light))
        assert width <= own_width + 0.2, (furrows, seed, width, own_width)
        assert peak >= 0.9 * own_peak, (furrows, seed, peak, own_peak)


def _width_across(S, direction, n=120):
    """Full width at half maximum (px) of the highlight across a crest
    through the middle of the canvas in ``direction``, averaged along it
    over its middle half (sampled every 1/8 px, bilinear)."""
    tx, ty = direction
    lat = np.arange(-8, 8, 0.125)
    p = np.zeros(len(lat))
    for s in np.arange(-n / 4, n / 4, 0.5):
        p += _bilinear(S, n / 2 + s * tx - lat * ty, n / 2 + s * ty + lat * tx)
    i = int(np.argmax(p))
    above = np.nonzero(p > 0.5 * p[i])[0]
    return (above[-1] - above[0] + 1) * 0.125


def test_a_slanting_glint_is_about_as_narrow_as_a_level_one():
    # the light moves along the crest, by at most half a pixel across it
    level = _width_across(_highlight(*_furrowed_ridge(0, furrows=0.0)[:2]), (1, 0))
    for angle in (22, 30, 45, 60):
        c, light, along = _furrowed_ridge(angle, furrows=0.0)
        width = _width_across(_highlight(c, light), along)
        assert width <= level + 0.5, (angle, width, level)


def _dome(sigma, top, n=64):
    c = _layer(
        Canvas(n, n, surface=surfaces.GLASS.with_(transparent=False)),
        Paint("#3060a0"),
        4.0,
    )
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
    h = (top * np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)).astype(np.float32)
    c.surface_height = lambda relief=1.0: h  # type: ignore[method-assign]
    return c


def test_domes_and_round_dabs_keep_their_own_highlight(monkeypatch):
    # nothing is gathered on round paint: the curvature runs no one way
    from oilpaint import canvas as cm

    def both(c, light):
        lin = c._render_linear()
        L = np.asarray(light, float) / np.linalg.norm(light)
        with monkeypatch.context() as m:
            m.setattr(cm, "GLOSS_SPAN", 0.0)
            own = c._light(lin, L, 1.0, 1.0)
        return c._light(lin, L, 1.0, 1.0), own

    for sigma, top in ((1.5, 4.0), (4.0, 6.0), (8.0, 10.0)):
        for light in ((0.0, -1.0, 0.6), RAKING):
            got, own = both(_dome(sigma, top), light)
            assert np.abs(got - own).max() < 1.5 / 255, (sigma, light)
    c = Canvas(300, 200, seed=4)
    for i in range(12):
        x, y = 30 + (i % 6) * 45, 50 + (i // 6) * 90
        c.stroke(
            [(x, y), (x + 1.5, y + 1.0)],
            P.CADMIUM_RED,
            amount=40,
            pressure=0.8,
            brush=brushes.ROUND_HOG.sized(14 + i),
            seed=i,
        )
    for light in ((-0.6, -0.8, 1.1), RAKING):
        got, own = both(c, light)
        gloss = (
            got
            - c._light(
                c._render_linear(), np.asarray(light) / np.linalg.norm(light), 1.0, 0.0
            )
        ).mean(-1)
        lit, own_lit = (got - own).mean(-1) + gloss, gloss
        assert abs(got.max() - own.max()) < 1e-6
        a, b = (lit >= 8 / 255).sum(), (own_lit >= 8 / 255).sum()
        assert abs(a - b) <= 0.03 * b, (light, a, b)


def test_a_row_of_highlight_spreads_to_exactly_its_neighbourhood():
    from oilpaint import _render

    rng = np.random.default_rng(0)
    for W in (1, 2, 7, 40):
        src = np.where(rng.random(W) < 0.1, rng.random(W), 0.0)
        for r in (0, 1, 3):
            out = np.empty(W, np.uint8)
            _render._near_row(src, r, out)
            want = [(src[max(x - r, 0) : x + r + 1] != 0).any() for x in range(W)]
            assert out.tolist() == [int(v) for v in want], (W, r)


def _own_and_gathered(c, light, monkeypatch, **const):
    """The canvas lit (black paint: the highlight alone, rolled off) with
    the highlight gathered, and each pixel's own (GLOSS_SPAN 0)."""
    from oilpaint import canvas as cm

    black = np.zeros((c.height, c.width, 3), np.float32)
    with monkeypatch.context() as m:
        for k, v in const.items():
            m.setattr(cm, k, v)
        got = c._light(black, light, 1.0, 1.0)
        m.setattr(cm, "GLOSS_SPAN", 0.0)
        own = c._light(black, light, 1.0, 1.0)
    return got, own


def test_far_from_any_highlight_nothing_changes(monkeypatch):
    # a pixel takes highlight only from within the span (and the pixel next
    # to it); elsewhere the light is exactly the ungathered light
    from oilpaint import canvas as cm

    c, light, _ = _furrowed_ridge(30, 1, furrows=0.0)
    got, own = _own_and_gathered(c, light, monkeypatch)
    lit = own.max(-1) > 0
    reach = int(np.ceil(cm.GLOSS_SPAN)) + 1
    near = np.zeros_like(lit)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            near |= np.roll(np.roll(lit, dy, 0), dx, 1)
    assert near.mean() < 0.5 and np.array_equal(got[~near], own[~near])
    # and with nothing gathered by the gate, all is exactly as ungathered
    shut, own = _own_and_gathered(
        c, light, monkeypatch, CREST_GATE=(2.0, 3.0, 1e9, 2e9)
    )
    assert np.array_equal(shut, own)


def test_paint_too_thin_to_shine_takes_no_highlight_from_its_neighbours():
    # gathered light is shown as glossy as the pixel's own film lets it
    from oilpaint import canvas as cm

    c, light, _ = _furrowed_ridge(30, 1, furrows=0.0)
    n = c.width
    film = np.full((n, n), 4.0, np.float32)
    film[:, n // 2 :] = cm.GLOSS_FILM[0] + 0.05  # barely a film: almost no gloss
    _layer(c, Paint("#3060a0"), film)
    S = _highlight(c, light)
    assert S[:, : n // 2].max() > 50
    assert S[:, n // 2 :].max() < 1.0


def test_bowls_planes_and_saddles_keep_their_highlight_up_to_the_border(
    monkeypatch,
):
    # at the border the curvature stencils would run off the canvas and
    # read a slope as a crest: nothing is gathered there
    n = 40
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
    for h in (
        0.012 * (xx**2 + yy**2),  # a bowl
        0.3 * xx + 0.4 * yy,  # a plane, near the slope that mirrors RAKING
        0.012 * (xx**2 - yy**2),  # a saddle
    ):
        c = _layer(
            Canvas(n, n, surface=surfaces.GLASS.with_(transparent=False)),
            Paint("#3060a0"),
            4.0,
        )
        hh = h.astype(np.float32)
        c.surface_height = lambda relief=1.0, hh=hh: hh  # type: ignore[method-assign]
        lit = 0
        for light in ((0.0, -1.0, 0.6), RAKING, (0.8, 0.3, 0.6)):
            L = np.asarray(light, float) / np.linalg.norm(light)
            got, own = _own_and_gathered(c, L, monkeypatch)
            lit += int((own.max(-1) > 0.01).sum())
            assert np.array_equal(got, own), light
        assert lit > 20, lit  # there is a highlight to keep


def test_gathered_light_takes_the_receiving_pixels_mattness(monkeypatch):
    from oilpaint import _render
    from oilpaint import canvas as cm

    c, light, _ = _furrowed_ridge(30, 1)
    black = np.zeros((c.height, c.width, 3), np.float32)
    matte = np.zeros((c.height, c.width), np.float32)
    matte[:, c.width // 2 :] = 0.5

    def spec(m):
        v = c._light(black, light, 1.0, 1.0, m)[..., 0].astype(np.float64)
        return v / np.sqrt(1 - v * v)

    full, half = spec(None), spec(matte)
    keep = 1 - _render.MATTE_GLOSS_LOSS * 0.5
    right = np.s_[:, c.width // 2 :]
    assert np.abs(half[right] - keep * full[right]).max() < 1e-5 * full.max()
    assert np.array_equal(half[:, : c.width // 2], full[:, : c.width // 2])
    # and gloss 0 is the ungathered render, bit for bit, on painted paint
    p = Canvas(120, 80, seed=2)
    p.stroke([(10, 60), (110, 20)], P.CADMIUM_RED, amount=80, width=30, seed=3)
    with monkeypatch.context() as m:
        m.setattr(cm, "GLOSS_SPAN", 0.0)
        want = p.render(light=RAKING, gloss=0.0)
    assert np.array_equal(p.render(light=RAKING, gloss=0.0), want)


def _gathered_by_hand(own, h, span, taps, gate):
    """The highlight gathered along crests as shade defines it (_crest,
    _gather, _tap), written out plainly, from each pixel's own highlight
    ``own`` (where the lamp's share, the film and mattness are the same at
    every pixel)."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]

    def at(dy, dx):
        return h[np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)]

    h64 = h.astype(np.float64)
    hxy = 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)).astype(np.float64)
    wxx = 0.25 * (at(0, 2) - 2.0 * h64 + at(0, -2))
    wyy = 0.25 * (at(2, 0) - 2.0 * h64 + at(-2, 0))
    k = np.array([1, 4, 6, 4, 1]) / 16

    def blur(a):
        rows = np.zeros_like(a)
        for i in range(5):
            rows += k[i] * a[:, np.clip(np.arange(W) + i - 2, 0, W - 1)]
        out = np.zeros_like(a)
        for i in range(5):
            out += k[i] * rows[np.clip(np.arange(H) + i - 2, 0, H - 1)]
        return out

    sxx, sxy, syy = (
        blur(wxx**2 + hxy**2),
        blur(hxy * (wxx + wyy)),
        blur(wyy**2 + hxy**2),
    )
    d = np.sqrt((sxx - syy) ** 2 + 4 * sxy**2)
    l1 = 0.5 * (sxx + syy + d)
    c1, c2 = np.sqrt(l1), np.sqrt(np.maximum(0.5 * (sxx + syy - d), 0))

    def smooth(e0, e1, v):
        u = np.clip((v - e0) / (e1 - e0), 0, 1)
        return u * u * (3 - 2 * u)

    aniso = np.where(c1 > 0, (c1 - c2) / np.where(c1 > 0, c1 + c2, 1), 0)
    reach = span * smooth(gate[0], gate[1], aniso) * smooth(gate[2], gate[3], c1)
    tx = np.where(sxx >= syy, -sxy, -(l1 - sxx))
    ty = np.where(sxx >= syy, l1 - syy, sxy)
    norm = np.hypot(tx, ty)
    reach = np.where(norm > 0, reach, 0.0)
    edge = 4  # nothing gathered this near the border
    inside = (yy >= edge) & (yy < H - edge) & (xx >= edge) & (xx < W - edge)
    reach = np.where(inside, reach, 0.0)
    tx, ty = tx / np.where(norm > 0, norm, 1), ty / np.where(norm > 0, norm, 1)
    out = own.copy()
    for y, x in zip(*np.nonzero(reach > 1e-3), strict=True):
        total = weights = 0.0
        for kk in range(-taps, taps + 1):
            w = 1.0 - abs(kk) / (taps + 1.0)
            o = reach[y, x] * kk / taps
            fx = min(max(x + o * tx[y, x], 0.0), W - 1.0)
            fy = min(max(y + o * ty[y, x], 0.0), H - 1.0)
            if abs(tx[y, x]) >= abs(ty[y, x]):
                j, i = min(int(fy + 0.5), H - 1), int(fx)
                a = fx - i
                v = (1 - a) * own[j, i] + a * own[j, min(i + 1, W - 1)]
            else:
                i, j = min(int(fx + 0.5), W - 1), int(fy)
                a = fy - j
                v = (1 - a) * own[j, i] + a * own[min(j + 1, H - 1), i]
            total += w * v
            weights += w
        out[y, x] = total / weights
    return out


def test_the_highlight_is_its_own_averaged_along_the_crest(monkeypatch):
    # the kernel against the definition, on a furrowed slanting crest under
    # a high lamp (no cast shadows: every pixel sees all of it)
    from oilpaint import _render
    from oilpaint import canvas as cm

    c, _, _ = _furrowed_ridge(30, 2, n=64)
    a = np.radians(30)
    light = np.array([0.3 * np.sin(a), -0.3 * np.cos(a), 1.0])
    light /= np.linalg.norm(light)
    h = c.surface_height()
    vis = np.empty_like(h)
    lx, ly, lz = (float(v) for v in light)
    _render.shadows(h, lx, ly, lz, cm.LIGHT_SIZE, cm.SHADOW_REACH, vis)
    assert vis.min() == 1.0
    got, own = _own_and_gathered(c, light, monkeypatch)

    def unroll(v):  # the roll-off towards white, undone (black paint)
        v = v[..., 0].astype(np.float64)
        return v / np.sqrt(1 - v * v)

    own_spec = unroll(own)
    gy, gx = np.gradient(h.astype(np.float64))
    facing = light[2] - gx * light[0] - gy * light[1] > 0  # (all but the edges)
    want = _gathered_by_hand(own_spec, h, cm.GLOSS_SPAN, cm.GLOSS_TAPS, cm.CREST_GATE)
    inner = np.zeros(h.shape, bool)
    inner[2:-2, 2:-2] = True
    sel = inner & facing
    assert own_spec.max() > 0.05 and (want != own_spec)[sel].mean() > 0.05
    assert np.abs(unroll(got) - want)[sel].max() < 1e-5 * own_spec.max()


# --------------------------------------------------------------------------
# cast shadows against a brute-force march


def _brute_shadows(h, L, light_size, max_reach):
    """``_render.shadows`` the slow way: every ray of the lamp marched one
    pixel at a time to the end of its reach, nothing skipped."""
    from oilpaint import _render as R

    H, W = h.shape
    L = np.asarray(L, float) / np.linalg.norm(L)
    horiz = np.hypot(L[0], L[1])
    elev = np.arctan2(L[2], horiz)
    tan_lo = np.tan(max(elev - light_size, R.ELEV_MIN))
    tan_hi = np.tan(min(elev + light_size, R.ELEV_MAX))
    need = (h.max() - h.min()) / tan_lo + 1.0
    reach = min(need, max_reach)
    fade_from = reach * (1 - R.FADE) if need > max_reach else reach + 1
    az = np.arctan2(L[1], L[0])
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    h64 = h.astype(np.float64)
    vis = np.zeros((H, W))
    for i in range(R.N_RAYS):
        a = az + light_size * (i - (R.N_RAYS - 1) / 2) / ((R.N_RAYS - 1) / 2)
        dx, dy = np.cos(a), np.sin(a)
        best = np.full((H, W), -1e9)
        t = 1.0
        while t <= reach:
            fx, fy = xs + t * dx, ys + t * dy
            inside = (fx >= 0) & (fy >= 0) & (fx <= W - 1) & (fy <= H - 1)
            if not inside.any():
                break
            fxc, fyc = np.clip(fx, 0, W - 1), np.clip(fy, 0, H - 1)
            ix, iy = fxc.astype(int), fyc.astype(int)
            ax, ay = fxc - ix, fyc - iy
            ix1, iy1 = np.minimum(ix + 1, W - 1), np.minimum(iy + 1, H - 1)
            hs = (1 - ay) * ((1 - ax) * h64[iy, ix] + ax * h64[iy, ix1]) + ay * (
                (1 - ax) * h64[iy1, ix] + ax * h64[iy1, ix1]
            )
            tan = (hs - h64) / t
            if t > fade_from:
                u = min((t - fade_from) / (reach - fade_from), 1.0)
                tan = tan * (1 - u * u * (3 - 2 * u))
            best = np.where(inside, np.maximum(best, tan), best)
            t += 1.0
        u = np.clip((best - tan_lo) / (tan_hi - tan_lo), 0, 1)
        vis += np.where(best > tan_lo, 1 - u * u * (3 - 2 * u), 1.0)
    vis /= R.N_RAYS
    p = np.pad(h64, 1, mode="edge")  # central differences, edges clamped
    gx = 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    gy = 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    vis[L[2] - gx * L[0] - gy * L[1] <= 0] = 1.0
    return vis


def _adversarial_fields(S=96):
    rng = np.random.default_rng(0)
    fields = {}
    f = np.zeros((S, S), np.float32)
    for _ in range(25):
        f[rng.integers(0, S), rng.integers(0, S)] = rng.uniform(3, 30)
    fields["single-pixel spikes"] = f
    yy, xx = np.mgrid[0:S, 0:S]
    d = np.abs((xx - S / 2) * 0.37 + (yy - S / 2) * 0.93)
    fields["1-px oblique wall"] = np.where(d < 0.6, 20.0, 0.0).astype(np.float32)
    f = np.zeros((S, S), np.float32)
    f[30:66, 30:66] = 15.0
    fields["plateau"] = f
    f = np.zeros((S, S), np.float32)
    f[:, 0] = f[0, :] = f[:, -1] = f[-1, :] = 25.0
    fields["ridges along the image edges"] = f
    f = (rng.random((S, S)) * 1.0).astype(np.float32)
    for _ in range(15):
        f[rng.integers(0, S), rng.integers(0, S)] += rng.uniform(5, 40)
    fields["noise and spikes"] = f
    f = np.zeros((70, 131), np.float32)  # not square
    f[20:40, 50:53] = 18.0
    f[60, 100] = 25.0
    fields["non-square"] = f
    f = np.zeros((9, 11), np.float32)  # smaller than a tile
    f[4, 5] = 6.0
    f[0, :] = 4.0
    fields["tiny"] = f
    return fields


@pytest.mark.parametrize(
    "light",
    [
        (-0.6, -0.8, 0.7),
        (0.37, -0.93, 0.2),
        (-0.71, 0.71, 0.05),
        (0.2, 0.98, 1.5),
        (-1.0, 0.0, 0.5),  # along the rows
        (0.0, 1.0, 0.4),  # along the columns
    ],
)
def test_cast_shadows_match_a_brute_force_march(light):
    from oilpaint import _render as R
    from oilpaint.canvas import LIGHT_SIZE

    for name, h in _adversarial_fields().items():
        for max_reach in (400.0, 30.0):  # the whole way, and cut short (fading)
            L = np.asarray(light, float) / np.linalg.norm(light)
            got = np.empty_like(h)
            R.shadows(h, L[0], L[1], L[2], LIGHT_SIZE, max_reach, got)
            want = _brute_shadows(h, L, LIGHT_SIZE, max_reach)
            assert (want < 0.99).any(), (name, "casts no shadow to compare")
            assert np.abs(got - want).max() < 1e-4, (name, max_reach)
