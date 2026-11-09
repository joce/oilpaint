"""Rendering: the colour of the paint, and the light on its relief."""

import os
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
