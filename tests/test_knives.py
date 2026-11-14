"""Knives: slabs, scraping, dragging wet paint, combs and lifting off."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, Paint, brushes
from oilpaint import _engine as eng
from oilpaint import palette as P
from oilpaint._engine import PACKED
from oilpaint.brush import _VOLUME_PER_AMOUNT, _pack


def _knife_slab(pressure, **kw):
    c = Canvas(300, 120, seed=2)
    c.stroke(
        [(40, 60), (200, 60)],
        paint=P.TITANIUM_WHITE,
        amount=60,
        brush=brushes.PALETTE_KNIFE.sized(50),
        pressure=pressure,
        seed=2,
        **kw,
    )
    return c


def test_knife_pressure_sets_how_thick_the_slab_is():
    light, firm, scraped = (_knife_slab(p) for p in (0.2, 0.6, 0.95))
    core = (slice(45, 75), slice(70, 130))
    t = [c.thickness[core].mean() for c in (light, firm, scraped)]
    assert t[0] > 2 * t[1] > 4 * t[2]
    slab = light.thickness[core]
    assert slab.std() / slab.mean() < 0.12  # a flat, smooth slab
    # scraped close, paint is left in the valleys of the weave only
    weave = scraped.grain[core]
    assert np.corrcoef(scraped.thickness[core].ravel(), weave.ravel())[0, 1] < -0.8


def test_a_knife_runs_out_cleanly():

    c = Canvas(400, 120, seed=4)
    c.stroke(
        [(40, 60), (360, 62)],
        brush=Brush(brushes.KNIFE_TROWEL).load(P.CADMIUM_YELLOW, 30),
        pressure=0.35,
        seed=60,
    )
    thick = np.nonzero(c.thickness.max(axis=0) > 2.0)[0]
    assert 150 < thick.max() < 340  # the slab ends before the stroke does...
    # ...thinning out, without leaving rails behind
    assert c.thickness[:, thick.max() + 10 :].max() < 3.0


def test_a_knife_pushes_wet_paint_into_ridges_beside_it():
    c = Canvas(300, 120, seed=3)
    c.stroke(
        [(20, 60), (280, 60)],
        paint=P.ULTRAMARINE_BLUE,
        amount=40,
        brush=brushes.FLAT_HOG.sized(80),
        seed=3,
    )
    before = c.thickness.copy()
    total = before.sum()
    knife = c.stroke(
        [(60, 60), (240, 60)],
        brush=brushes.PALETTE_KNIFE.sized(40),
        angle=90,
        pressure=0.95,
        seed=4,
    )
    after = c.thickness
    cols = slice(100, 200)
    # scraped down, to about half (how far depends a little on where the hog's
    # furrows ran: the blade rests on the high points around it)
    assert after[55:66, cols].mean() < 0.6 * before[55:66, cols].mean()
    beside = np.r_[48:52, 67:71]
    # piled up beside
    assert after[beside, cols].mean() > before[beside, cols].mean() + 0.3
    assert after.sum() + knife.volume == pytest.approx(total, rel=1e-4)


def test_a_comb_scores_grooves():
    c = Canvas(300, 120, seed=3)
    c.stroke(
        [(20, 60), (280, 60)],
        paint=P.SAP_GREEN,
        amount=14,
        brush=brushes.FLAT_HOG.sized(70),
        seed=3,
    )
    before = c.thickness[40:80, 100:200].mean(axis=1)
    c.stroke([(30, 60), (270, 60)], brush=brushes.KNIFE_COMB, pressure=0.6, seed=4)
    after = c.thickness[40:80, 100:200].mean(axis=1)
    assert after.std() > 1.3 * before.std() and after.min() < 0.85 * before.min()


def test_lifting_a_knife_pulls_up_peaks():
    def end_texture(lift):  # streaks pulled along the stroke: texture across it
        c = _knife_slab(0.4, lift=lift)
        return float(np.abs(np.diff(c.thickness[45:75, 190:215], axis=0)).mean())

    assert end_texture(1.0) > 1.5 * end_texture(0.0)


def test_paint_is_conserved_with_knives_and_squeezing():
    c = Canvas(300, 200, seed=5)
    loaded = 0.0
    for k, (b, amount, pressure) in enumerate(
        [
            (brushes.PALETTE_KNIFE, 40, 0.3),
            (brushes.KNIFE_PEAR, 30, 0.8),
            (brushes.FLAT_HOG.sized(30), 50, 0.9),
        ]
    ):
        br = c.stroke(
            [(60, 50 + 50 * k), (240, 60 + 40 * k)],
            paint=P.ALL[k],
            amount=amount,
            brush=b,
            pressure=pressure,
            seed=k,
        )
        loaded += amount * _VOLUME_PER_AMOUNT * br.width**2 - br.volume
    knife = c.stroke(
        [(150, 20), (150, 180)], brush=brushes.KNIFE_STRAIGHT, pressure=0.9, seed=9
    )
    loaded -= knife.volume
    assert c.thickness.sum() == pytest.approx(loaded, rel=1e-4)


def test_knife_factory():
    k = brushes.knife("pear", 30, flex=0.8)
    assert (
        k.rigid
        and k.shape == "pear"
        and k.width == 30
        and k.stiffness == pytest.approx(0.6)
    )
    assert brushes.knife("straight", 40, teeth=6).clumps == 6
    assert not brushes.FLAT_HOG.rigid
    with pytest.raises(ValueError):
        brushes.knife("sword", 30)
    with pytest.raises(ValueError):
        brushes.knife("pear", 30, flex=2)


def test_impasto_casts_shadows_under_a_raking_light():
    c = _knife_slab(0.2)
    lit = c.render(light=(-0.6, -0.8, 0.5))
    flat = c.render(lighting=False)
    # the slab's far edge (away from the light) throws a shadow onto the canvas beside
    # it
    shade = (lit[80:95, 70:130].mean(-1) / flat[80:95, 70:130].mean(-1)).mean(axis=1)
    assert shade.min() < 0.7
    assert c.surface_height().max() > 4.0


def test_a_dry_knife_dragged_across_white_on_purple():
    purple = Paint.mix(
        (P.DIOXAZINE_PURPLE, 2), (P.QUINACRIDONE_MAGENTA, 1), (P.TITANIUM_WHITE, 1)
    )

    def layered():
        c = Canvas(360, 300, seed=4)
        for y in range(50, 250, 34):
            c.stroke(
                [(30, y), (330, y + 4)],
                paint=purple,
                amount=60,
                brush=brushes.KNIFE_TROWEL.sized(56),
                pressure=0.3,
                seed=y,
            )
        for y in range(80, 230, 34):
            c.stroke(
                [(60, y), (300, y - 3)],
                paint=P.TITANIUM_WHITE,
                amount=60,
                brush=brushes.KNIFE_TROWEL.sized(50),
                pressure=0.35,
                seed=1000 + y,
            )
        return c

    def lightness(c, region):
        return float(c.render(lighting=False)[region].mean())

    swipe = (slice(90, 200), slice(165, 195))
    c = layered()
    before_t, before_l = c.thickness[swipe].mean(), lightness(c, swipe)
    # wet white laid over wet purple stays white on top
    assert before_l > 0.9
    c.stroke(
        [(180, 20), (175, 150), (185, 285)],
        brush=brushes.PALETTE_KNIFE.sized(60),
        pressure=0.6,
        seed=7,
    )
    # a firm drag smears the white; it doesn't turn lavender
    assert lightness(c, swipe) > before_l - 0.05
    assert c.thickness[swipe].mean() < before_t
    c = layered()
    c.stroke(
        [(180, 20), (175, 150), (185, 285)],
        brush=brushes.PALETTE_KNIFE.sized(60),
        pressure=0.95,
        seed=7,
    )
    # pressed hard, it cuts a channel through both layers (some of the dragged mixture
    # stays in it)
    assert c.thickness[swipe].mean() < 0.6 * before_t


def test_knife_slabs_fill_on_curves_and_in_dabs():
    c = Canvas(400, 200, seed=1)
    c.stroke(
        [(60, 60), (220, 60), (240, 120)],
        paint=P.CADMIUM_RED,
        amount=40,
        brush=brushes.KNIFE_ANGLED,
        pressure=0.4,
        seed=21,
    )
    c.stroke(
        [(320, 100)],
        paint=P.TITANIUM_WHITE,
        amount=30,
        brush=brushes.KNIFE_SPOON,
        pressure=0.3,
        seed=22,
    )
    # a solid slab, not an outline
    assert c.thickness[50:70, 100:180].min() > 2.0
    assert c.thickness[95:105, 315:325].min() > 1.0


def test_a_dry_knife_drags_white_out_over_purple():
    purple = Paint.mix(
        (P.DIOXAZINE_PURPLE, 2), (P.QUINACRIDONE_MAGENTA, 1), (P.TITANIUM_WHITE, 1)
    )
    c = Canvas(500, 200, seed=4)
    for y in range(40, 180, 30):
        c.stroke(
            [(20, y), (480, y + 3)],
            paint=purple,
            amount=200,
            brush=brushes.KNIFE_STRAIGHT.sized(60),
            pressure=0.35,
            seed=y,
        )
    for y in range(70, 150, 30):  # white over the left part, ending at x = 230
        c.stroke(
            [(40, y), (230, y - 2)],
            paint=P.TITANIUM_WHITE,
            amount=90,
            brush=brushes.KNIFE_STRAIGHT.sized(56),
            pressure=0.4,
            seed=1000 + y,
        )
    before = c.render(lighting=False).mean(-1)
    total = c.thickness.sum()
    knife = c.stroke(
        [(100, 100), (250, 103), (420, 100)],
        brush=brushes.PALETTE_KNIFE.sized(60),
        pressure=0.6,
        seed=8,
    )
    after = c.render(lighting=False).mean(-1)
    band = slice(90, 112)

    def gain(x0):
        return float(
            after[band, x0 : x0 + 20].mean() - before[band, x0 : x0 + 20].mean()
        )

    # the white is dragged out over the purple, in a tail that fades with distance
    assert gain(240) > 0.15 and gain(240) > gain(300) > gain(380) > -0.02
    # untouched
    assert (
        abs(float(after[150:165, 260:300].mean() - before[150:165, 260:300].mean()))
        < 0.01
    )
    assert c.thickness.sum() + knife.volume == pytest.approx(total, rel=2e-3)


def _wet_knife_field(n=6, width=600, seed=4):
    purple = Paint.mix(
        (P.DIOXAZINE_PURPLE, 2), (P.QUINACRIDONE_MAGENTA, 1), (P.TITANIUM_WHITE, 1)
    )
    c = Canvas(width, 200, seed=seed)
    for k in range(n):
        y = 30 + 30 * k
        c.stroke(
            [(20, y), (width - 20, y + 3)],
            paint=purple,
            amount=200,
            brush=brushes.KNIFE_STRAIGHT.sized(60),
            pressure=0.35,
            seed=k + seed - 3,
        )
    return c


def _ripple_along_x(surface, rows, cols):
    band = surface[rows, cols].mean(axis=0)
    return float(
        (band - np.convolve(band, np.ones(21) / 21, mode="same"))[15:-15].std()
    )


def test_overlapping_wet_knife_slabs_are_smooth():
    # a knife laid over the edge of a wet slab doesn't leave a bead at every step
    # (the blade's ends now press paint out into a ridge along the edge of the
    # slab, which comes and goes along it: the edge isn't flat, but nothing in
    # it repeats at the step, about 7.9 px)
    rough, at_step = [], []
    for seed in (4, 5, 6):
        c = _wet_knife_field(n=2, seed=seed)
        surface = (c._relief().floor + c.top_v + c.und_v)[86:100, 120:480].astype(float)
        ripple = surface - np.apply_along_axis(
            lambda r: np.convolve(r, np.ones(15) / 15, mode="same"), 1, surface
        )
        ripple = ripple[:, 10:-10]
        rough.append(ripple.std())
        band = ripple.mean(0)
        phase = ((np.arange(130, 470) + 0.5 - 20) % 7.9) / 7.9
        bins = np.minimum((phase * 8).astype(int), 7)
        at_step.append(np.std([band[bins == k].mean() for k in range(8)]))
    assert np.mean(rough) < 0.15  # (it was 0.11 with flat slab edges)
    assert np.mean(at_step) < 0.035  # (0.002-0.015 with flat slab edges)


def test_a_clean_knife_drag_leaves_no_ribs():
    c = _wet_knife_field()
    c.stroke(
        [(100, 100), (300, 103), (500, 100)],
        brush=brushes.PALETTE_KNIFE.sized(60),
        pressure=0.6,
        seed=8,
    )
    surface = c._relief().floor + c.top_v + c.und_v
    assert _ripple_along_x(surface, slice(90, 112), slice(150, 450)) < 0.3


def test_moved_paint_is_fully_covered():
    # paint dragged or merged into a pixel must not leave it looking part-covered (light
    # specks)
    c = _wet_knife_field(n=3)
    lum = c.render(lighting=False).mean(-1)
    core = lum[50:105, 60:540]
    assert core.max() - np.median(core) < 0.1


def test_colour_bands_wander_on_a_knife_but_not_on_a_brush():

    def boundary_wander(tool, seed):
        c = Canvas(500, 160, seed=seed + 2)
        br = (
            Brush(tool)
            .load(P.CADMIUM_RED, 30, across="left")
            .load(P.TITANIUM_WHITE, 30, across="right")
        )
        c.stroke([(40, 80), (460, 80)], brush=br, pressure=0.4, seed=seed)
        img = c.render(lighting=False)
        # red minus blue: red side vs white side
        redness = img[40:120, 80:300, 0] - img[40:120, 80:300, 2]
        # where the join between red and white lies in each column (counting
        # only the middle of the stroke: its outer edges are ragged)
        red_rows = (redness[25:55] > 0.5 * redness.max()).sum(axis=0)
        return float(np.std(red_rows))

    knife = np.mean(
        [boundary_wander(brushes.PALETTE_KNIFE.sized(60), s) for s in (1, 2, 3)]
    )
    brush = np.mean(
        [boundary_wander(brushes.FLAT_SABLE.sized(60), s) for s in (1, 2, 3)]
    )
    assert knife > 1.0
    assert brush < 0.6 * knife


def test_a_scraper_leaves_no_outline_on_bare_canvas():
    c = Canvas(500, 260, seed=21)
    c.stroke(
        [(20, 130), (480, 130)],
        paint=P.ULTRAMARINE_BLUE,
        amount=16,
        brush=brushes.FLAT_HOG.sized(90),
        seed=32,
    )
    top_edge = int(np.nonzero((c.top_v + c.und_v)[:, 250] > 0.05)[0].min())
    c.stroke(
        [(180, 70), (320, 190)],
        brush=brushes.SCRAPER,
        angle=-30,
        pressure=0.95,
        seed=33,
    )
    wet = c.top_v + c.und_v
    # the scraper moves down and away: nothing of the band ends up on the bare canvas
    # above it
    assert wet[: top_edge - 4].sum() < 5.0
    # and no light pinholes inside the scraped paint
    lum = c.render(lighting=False).mean(-1)[:, 150:400]
    w = wet[:, 150:400]
    inside = (w > 0.5) & (np.roll(w, 2, 0) > 0.5) & (np.roll(w, -2, 0) > 0.5)
    assert (lum[inside] > 0.6).sum() == 0


def _nudged(v, k):
    for _ in range(abs(k)):
        v = float(np.nextafter(v, np.inf if k > 0 else -np.inf))
    return v


def test_a_knife_dab_at_a_whole_number_position_presses_out_the_same_paint():
    # Pressed straight down, the knife's centre lies on a pixel's middle: the
    # paint pressed out at its ends was smeared along the stroke or not as
    # rounding left that point a hair before or past the end of the stroke.
    def dab(x, y):
        c = Canvas(120, 100, seed=7)
        knife = Brush(brushes.PALETTE_KNIFE).load(P.CADMIUM_RED, 9)
        c.stroke([(x, y)], brush=knife, pressure=0.45, width=2.6, seed=3)
        return c

    plain = dab(60.0, 50.0)
    for kx, ky in ((-1, -1), (1, 1), (-2, 1), (2, -2)):
        nudged = dab(_nudged(60.0, kx), _nudged(50.0, ky))
        a, b = plain.top.astype(float), nudged.top.astype(float)
        assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max(), (kx, ky)


def test_a_knife_going_back_over_its_slab_covers_the_same_at_pressure_0_3():
    # Below 0.3 a knife starts to skim: its bead lays paint, 1e-31 of it a
    # hair below. Passing back over its own slab it lays nothing else; it
    # spread its slab over more of each pixel only if it laid any, so the
    # cover there hung on that 1e-31.
    def back(p):
        c = Canvas(160, 120, seed=7)
        knife = Brush(brushes.PALETTE_KNIFE).load(P.TITANIUM_WHITE, 3.0)
        c.stroke([(30, 60), (120, 60), (40, 62)], brush=knife, pressure=p, seed=2)
        return c

    at, below = back(0.3), back(float(np.nextafter(0.3, 0.0)))
    assert np.abs(at.top_a.astype(float) - below.top_a).max() < 1e-6
    assert np.abs(at.top_v.astype(float) - below.top_v).max() < 1e-6


def test_paint_shoved_onto_its_own_kind_does_not_tie_with_the_film_over_it():
    # A knife pushes the wet layer below along over a thin film of another
    # colour: what it shoves is that layer's paint, as alike to the film as
    # the layer is. It goes on top whichever way rounding tips the two
    # likenesses (it joined the film, or covered it, as they fell).
    top = np.zeros((1, 1, PACKED), np.float32)
    und = np.zeros((1, 1, PACKED), np.float32)
    # (the film near enough to it in colour for the two to count as much
    # the same paint: else it would never join the film)
    film = Paint.mix((P.YELLOW_OCHRE, 4), (P.BURNT_UMBER, 1))
    top[0, 0] = _pack(film, 0.2)
    und[0, 0] = _pack(P.YELLOW_OCHRE, 2.0)
    joined = set()
    rng = np.random.default_rng(1)
    for _ in range(24):
        # (what is shoved is summed from single-precision paint: the same
        # paint, to rounding in its last bits)
        noise = 1.0 + rng.uniform(-1, 1, PACKED) * 2.0**-21
        pk = und[0, 0].astype(np.float64) * 0.37 * noise
        t = top.copy()
        joined.add(bool(eng._join_top_if_alike(t, und, pk, 0, 0)))
    assert joined == {False}


def test_a_knife_at_whole_number_positions_rests_on_the_same_pixels():
    # At whole-number positions the edge of the blade runs through the middle
    # of a row of pixels: each is half under it, exactly, and was counted
    # under it (holding it up) or not as rounding left that half.
    def drag(path, pressure, width):
        c = Canvas(200, 160, seed=7)
        for y in (20, 60, 100, 140):
            c.stroke(
                [(5, y), (100, y + 6), (195, y)],
                P.YELLOW_OCHRE,
                amount=14,
                brush=brushes.FLAT_HOG.sized(50),
                pressure=0.6,
            )
        c.stroke(
            path, brush=Brush(brushes.PALETTE_KNIFE), pressure=pressure, width=width
        )
        return c

    path, p, w = [(76.0, 114.0), (156.0, 114.0)], 0.78, 39.3
    plain = drag(path, p, w)
    for k in (-2, -1, 1, 2):
        nudged = drag(
            [(_nudged(x, k), _nudged(y, -k)) for x, y in path], _nudged(p, k), w
        )
        a, b = plain.top.astype(float), nudged.top.astype(float)
        assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max(), k
        np.testing.assert_array_equal(plain.top_id, nudged.top_id)


def test_paint_shoved_with_a_trace_of_the_film_in_it_joins_the_film():
    # Shoved paint holding a trace of the film (here a thousandth of it,
    # 1.4e-4 apart from the paint below in likeness, well clear of rounding)
    # is more like the film than the paint below is: it joins the film.
    top = np.zeros((1, 1, PACKED), np.float32)
    und = np.zeros((1, 1, PACKED), np.float32)
    top[0, 0] = _pack(Paint.mix((P.YELLOW_OCHRE, 4), (P.BURNT_UMBER, 1)), 0.2)
    und[0, 0] = _pack(P.YELLOW_OCHRE, 2.0)
    pk = und[0, 0].astype(np.float64) * 0.37 + top[0, 0].astype(np.float64) * 1e-3
    assert eng._join_top_if_alike(top.copy(), und, pk, 0, 0)
