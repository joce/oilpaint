import os
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import (
    Bezier,
    Brush,
    BrushType,
    Canvas,
    CatmullRom,
    Paint,
    Polyline,
    Stroke,
    Surface,
    brushes,
    mix_colors,
    surfaces,
)
from oilpaint import palette as P
from oilpaint import spectral as sp
from oilpaint.brush import COLUMNS

# --------------------------------------------------------------------------
# spectral / Kubelka-Munk


def test_rgb_reflectance_roundtrip():
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.15, 1.0, (200, 3))
    back = sp.reflectance_to_srgb(sp.linear_rgb_to_reflectance(sp.srgb_to_linear(cols)))
    assert np.abs(back - cols).max() < 1e-3  # < 0.3/255


def test_reflectance_is_physical():
    rng = np.random.default_rng(1)
    R = sp.linear_rgb_to_reflectance(rng.random((500, 3)))
    assert R.min() >= sp.MIN_REFLECTANCE and R.max() <= 1.0


def test_km_limits():
    ground = np.full(sp.N_BANDS, 0.6)
    K = np.linspace(0.1, 3.0, sp.N_BANDS)
    # no paint -> ground
    assert np.allclose(sp.km_layer(K * 0, 0.0, ground), ground)
    # pure dye -> Beer-Lambert, light passes the layer twice
    assert np.allclose(sp.km_layer(K, 0.0, ground), ground * np.exp(-2 * K), rtol=1e-6)
    # very thick -> masstone, independent of the ground
    assert np.allclose(
        sp.km_layer(K * 200, 400.0, ground), sp.km_opaque(K, 2.0), atol=1e-4
    )
    assert np.allclose(
        sp.km_layer(K * 200, 400.0, 0.0), sp.km_opaque(K, 2.0), atol=1e-4
    )


def test_solve_absorption_reproduces_target():
    target = sp.srgb_to_reflectance("#3a7bd5")
    for S in (0.0, 0.5, 10.0):
        K = sp.solve_absorption(target, S)
        assert np.allclose(sp.km_layer(K, S, 1.0), target, atol=1e-6)


# --------------------------------------------------------------------------
# paints and palette mixing


@pytest.mark.parametrize(
    "color", ["#c8201e", "#1b3fa0", "#f5d000", "#808080", "#ffffff"]
)
@pytest.mark.parametrize("opacity", [1.0, 0.3])
def test_paint_looks_like_its_color_on_white(color, opacity):
    paint = Paint(color, opacity=opacity)
    assert np.abs(paint.appearance() - sp.parse_color(color)).max() < 2e-3


def test_blue_and_yellow_make_green():
    r, g, b = sp.parse_color(mix_colors(["#1b3fa0", "#f5d000"]))
    assert g > r and g > b
    # whereas averaging RGB gives a grey-ish color with g < r
    avg = (sp.parse_color("#1b3fa0") + sp.parse_color("#f5d000")) / 2
    assert avg[1] < avg[0]


def test_mixing_a_paint_with_itself_changes_nothing():
    p = P.CADMIUM_RED
    m = Paint.mix((p, 1), (p, 3))
    assert np.allclose(m.K, p.K) and np.isclose(m.S, p.S)


def test_white_lightens_and_black_darkens():
    red = P.CADMIUM_RED
    tint = Paint.mix((red, 1), (P.TITANIUM_WHITE, 1))
    shade = Paint.mix((red, 1), (P.IVORY_BLACK, 1))

    def lum(c):
        return float(sp.LUMINANCE @ sp.srgb_to_linear(c))

    assert lum(tint.color) > lum(red.color) > lum(shade.color)


def test_glaze_is_transparent_body_color_is_not():
    over_black = "#000000"
    glaze = Paint.mix((P.ALIZARIN_CRIMSON, 1), (P.MEDIUM, 3))

    def lum(c):
        return float(sp.LUMINANCE @ sp.srgb_to_linear(c))

    # A thin glaze over black stays dark; an opaque light paint covers it.
    assert lum(glaze.appearance(1.0, over_black)) < 0.05
    assert lum(P.TITANIUM_WHITE.appearance(1.0, over_black)) > 0.7


# --------------------------------------------------------------------------
# paths


def test_catmull_rom_passes_through_points():
    pts = [(10, 10), (60, 40), (120, 20), (200, 80)]
    samples, _, _, knot = CatmullRom(pts).sample(0.5)
    for i, p in enumerate(pts):
        j = np.argmin(np.abs(knot - i))
        assert np.linalg.norm(samples[j] - p) < 1.0


def test_paths_resample_evenly():
    for path in (
        CatmullRom([(0, 0), (50, 80), (150, 20)]),
        Bezier([(0, 0), (40, 100), (120, -40), (160, 60)]),
        Polyline([(0, 0), (100, 0), (100, 100)]),
    ):
        pts, tan, u, _ = path.sample(2.0)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert steps.std() / steps.mean() < 0.1
        assert np.allclose(np.linalg.norm(tan, axis=1), 1.0)
        assert u[0] == 0 and u[-1] == 1


# --------------------------------------------------------------------------
# strokes on the canvas


def _brush_volume(brush):
    return brush.volume


def test_stroke_puts_paint_along_its_path_only():
    c = Canvas(200, 100, seed=0)
    c.stroke(
        [(20, 50), (100, 45), (180, 50)], paint="#1b3fa0", amount=10, width=20, seed=0
    )
    t = c.thickness
    assert t[45:52, 30:170].min() > 0.3  # covered along the spine
    assert t[:20].max() == 0 and t[85:].max() == 0  # nothing far away
    assert c.color_at(100, 46) != c.color_at(100, 5)


def test_paint_volume_is_conserved():
    c = Canvas(300, 200, seed=1)
    loaded = 0.0
    for i, (paint, amount) in enumerate(
        [("#1b3fa0", 8), ("#f5d000", 6), ("#c8201e", 5)]
    ):
        b = c.stroke(
            [(40, 40 + 50 * i), (150, 100), (260, 60 + 40 * i)],
            paint=paint,
            amount=amount,
            width=24,
            pressure=0.7,
            seed=i,
        )
        loaded += amount * 0.9 * b.width**2
        loaded -= _brush_volume(b)  # what stayed on the brush
    # clean brush, drags paint
    blender = c.stroke([(30, 100), (270, 110)], width=30, pressure=0.8)
    loaded -= _brush_volume(blender)
    assert c.thickness.sum() == pytest.approx(loaded, rel=1e-3)


def test_pressure_controls_wet_mixing():
    def crossing(pressure):
        c = Canvas(200, 160, seed=2)
        c.stroke(
            [(10, 80), (190, 80)], paint=P.ULTRAMARINE_BLUE, amount=12, width=40, seed=1
        )
        c.stroke(
            [(100, 5), (100, 155)],
            paint=P.CADMIUM_YELLOW,
            amount=12,
            width=34,
            pressure=pressure,
            seed=2,
        )
        return sp.parse_color(c.color_at(100, 80))

    light, hard = crossing(0.1), crossing(1.0)
    # Light touch leaves yellow on top; pressing hard churns in the blue (bluer,
    # darker).
    assert light[2] < hard[2] + 1e-6 or light.sum() > hard.sum()
    assert light[0] > hard[0]


def test_dirty_brush_carries_color():
    c = Canvas(300, 200, seed=3)
    c.stroke([(20, 60), (280, 60)], paint=P.CADMIUM_RED, amount=20, width=40, seed=1)
    brush = c.stroke(
        [(150, 10), (150, 120)], paint=P.TITANIUM_WHITE, amount=6, width=24, seed=2
    )
    c.stroke([(20, 170), (120, 170)], brush=brush, width=24, seed=3)
    r, _g, b = sp.parse_color(c.color_at(40, 170))
    w = P.TITANIUM_WHITE.color
    assert (r - b) > (w[0] - w[2]) + 0.03  # pinkish, not pure white


def test_dry_keeps_colors_and_stops_mixing():
    c = Canvas(160, 120, seed=4)
    c.stroke(
        [(10, 60), (150, 60)], paint=P.ULTRAMARINE_BLUE, amount=10, width=40, seed=1
    )
    before = c.render(lighting=False)
    c.dry()
    assert c.wet_thickness.max() == 0
    assert np.abs(c.render(lighting=False) - before).max() < 2e-3
    # A blending stroke over dry paint can no longer move it.
    c.stroke([(10, 60), (150, 60)], width=40, pressure=1.0)
    assert np.abs(c.render(lighting=False) - before).max() < 2e-3


def test_glaze_over_dry_paint_darkens_it():
    c = Canvas(160, 120, seed=5)
    c.stroke([(10, 60), (150, 60)], paint=P.CADMIUM_YELLOW, amount=10, width=50, seed=1)
    c.dry()
    yellow = sp.parse_color(c.color_at(80, 60))
    c.stroke(
        [(80, 5), (80, 115)],
        paint=Paint.mix((P.PHTHALO_BLUE, 1), (P.MEDIUM, 4)),
        amount=10,
        width=30,
    )
    glazed = sp.parse_color(c.color_at(80, 60))
    # yellow + blue glaze reads green
    assert glazed[1] > glazed[0] and glazed[1] > glazed[2]
    assert glazed.sum() < yellow.sum()


def test_dab_strokes_and_edges():
    c = Canvas(100, 100, seed=6)
    c.stroke([(50, 50)], paint="#ff0000", amount=10, width=20)
    assert c.thickness[50, 50] > 0.3
    # Strokes running off the canvas are fine and only use paint.
    b = c.stroke([(-50, 10), (150, 10)], paint="#00ff00", amount=4, width=20)
    assert np.isfinite(c.render()).all() and b.amount >= 0


def test_stroke_object_and_profiles():
    c = Canvas(200, 120, seed=7)
    c.paint(
        Stroke(
            path=Bezier([(10, 60), (60, 0), (140, 120), (190, 60)]),
            paint=P.VIRIDIAN,
            amount=12,
            width=[6, 30],
            pressure=lambda u: 0.2 + 0.6 * u,
            taper=0.2,
        )
    )
    assert c.thickness.sum() > 0


def test_brush_load_and_clean():
    b = Brush().load(P.CADMIUM_RED, amount=5, width=10)
    assert b.amount == pytest.approx(5)
    assert b.paint is not None and np.allclose(b.paint.K, P.CADMIUM_RED.K)
    b.clean()
    assert b.amount == 0 and b.paint is None


def test_save_png(tmp_path):
    c = Canvas(64, 48, seed=8)
    c.stroke([(5, 24), (60, 24)], paint="#336699", amount=5, width=12)
    path = tmp_path / "out.png"
    c.save(str(path))
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    try:
        from PIL import Image
    except ImportError:
        return
    img = np.asarray(Image.open(path))
    assert img.shape == (48, 64, 3)


def test_image_ground():
    img = np.zeros((40, 60, 3))
    img[:, :30] = [0.8, 0.3, 0.2]
    img[:, 30:] = [0.2, 0.4, 0.8]
    c = Canvas(60, 40, ground=img, texture=0)
    out = c.render(lighting=False)
    assert np.abs(out - img).max() < 2e-3


def test_canvas_survives_pickling():
    import pickle

    c = Canvas(60, 40, seed=9)
    c = pickle.loads(pickle.dumps(c))
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    assert c.top_v.max() > 0 and c.thickness.max() > 0


# --------------------------------------------------------------------------
# brush types


def _band_width(canvas, x, thresh=0.15):
    """Number of painted pixels in column x."""
    return int((canvas.thickness[:, x] > thresh).sum())


def _row_width(canvas, y, thresh=0.15):
    return int((canvas.thickness[y, :] > thresh).sum())


def test_default_brush_type_matches_default_stroke():
    a, b = Canvas(160, 80, seed=1), Canvas(160, 80, seed=1)
    a.stroke([(10, 40), (150, 40)], paint="#336699", amount=8, width=20, seed=3)
    b.stroke(
        [(10, 40), (150, 40)],
        paint="#336699",
        amount=8,
        width=20,
        seed=3,
        brush=BrushType(),
    )
    assert np.array_equal(a.thickness, b.thickness)


def test_flat_broadside_vs_edge_on():
    flat = brushes.FLAT_HOG.sized(30)
    wide, thin = Canvas(200, 100, seed=2), Canvas(200, 100, seed=2)
    wide.stroke([(20, 50), (180, 50)], paint="#aa3322", amount=20, brush=flat, angle=0)
    thin.stroke([(20, 50), (180, 50)], paint="#aa3322", amount=20, brush=flat, angle=90)
    assert 26 <= _band_width(wide, 100) <= 34
    assert _band_width(thin, 100) <= 0.35 * 30


def test_flat_has_square_ends_round_has_round_ends():
    def end_curvature(brush):
        c = Canvas(200, 100, seed=3)
        c.stroke(
            [(40, 50), (160, 50)], paint="#aa3322", amount=20, width=30, brush=brush
        )

        def first_x(y):
            return int(np.argmax(c.thickness[y] > 0.15))

        # how much later the stroke starts near its edge than along its centre
        return first_x(62) - first_x(50)

    assert end_curvature(brushes.FLAT_HOG) <= 2  # square end
    assert end_curvature(BrushType()) >= 4  # rounded end


def test_fixed_angle_calligraphy_thick_and_thin():
    # long side horizontal on the canvas
    nib = brushes.FLAT_SABLE.with_(angle=0, follow_path=False)
    c = Canvas(300, 300, seed=4)
    # moving along the nib
    c.stroke([(30, 60), (270, 60)], paint="#111111", amount=30, width=30, brush=nib)
    # moving across it
    c.stroke([(150, 100), (150, 290)], paint="#111111", amount=30, width=30, brush=nib)
    assert _band_width(c, 100) < 12  # horizontal stroke: thin
    assert _row_width(c, 200) > 24  # vertical stroke: wide


def test_pointed_round_thins_with_light_pressure():
    def width_at(p):
        c = Canvas(200, 100, seed=5)
        c.stroke(
            [(20, 50), (180, 50)],
            paint="#223355",
            amount=30,
            brush=brushes.ROUND_SABLE.sized(20),
            pressure=p,
        )
        return _band_width(c, 100)

    assert width_at(0.1) < 0.6 * width_at(0.9)


def test_angled_brush_paints_with_its_toe_first():
    c = Canvas(200, 100, seed=6)
    c.stroke(
        [(20, 50), (180, 50)],
        paint="#223355",
        amount=30,
        brush=brushes.DAGGER.sized(40),
        pressure=0.2,
    )
    col = c.thickness[:, 100]
    above, below = col[30:50].sum(), col[51:71].sum()
    assert max(above, below) > 2.5 * max(min(above, below), 1e-3)


def test_fan_leaves_separate_clumps():
    c = Canvas(200, 120, seed=7)
    c.stroke(
        [(20, 60), (180, 60)],
        paint="#335522",
        amount=30,
        brush=brushes.FAN_HOG.sized(60),
    )
    col = c.thickness[25:95, 100] > 0.1
    runs = np.count_nonzero(np.diff(col.astype(int)) == 1)
    assert runs >= 5


def test_palette_knife_is_smooth():
    c = Canvas(200, 100, seed=8)
    c.stroke(
        [(20, 50), (180, 50)],
        paint="#dddddd",
        amount=40,
        brush=brushes.PALETTE_KNIFE.sized(36),
    )
    core = c.thickness[40:60, 60:140]
    assert core.std() / core.mean() < 0.15


def test_every_preset_paints():
    for name, b in brushes.ALL.items():
        if not b.deposits:
            continue
        c = Canvas(120, 80, seed=9)
        c.stroke([(15, 40), (105, 35)], paint="#884422", amount=60, brush=b)
        assert c.thickness.max() > 0, name
        assert np.isfinite(c.render()).all(), name


def test_brush_factory_and_customisation():
    b = brushes.make("filbert", 30, hair="hog", length=0.2)
    assert b.shape == "filbert" and b.width == 30
    assert b.stiffness > brushes.make("filbert", 30, hair="sable").stiffness
    assert brushes.FLAT_HOG.sized(50).width == 50
    assert brushes.FLAT_HOG.with_(slant=0.4).slant == 0.4
    assert brushes.FLAT_HOG.width == 24  # presets are immutable
    with pytest.raises(ValueError):
        brushes.make("triangle")
    with pytest.raises(ValueError):
        brushes.make("flat", hair="nylon-ish")
    with pytest.raises(ValueError):
        BrushType(shape="hexagon")


def test_brush_object_keeps_its_type():
    brush = Brush(brushes.FLAT_HOG)
    brush.load("#aa3322", amount=20)
    c = Canvas(200, 100, seed=10)
    c.stroke([(20, 50), (180, 50)], brush=brush)
    assert 20 <= _band_width(c, 100) <= 28  # the flat's own 24 px width


# --------------------------------------------------------------------------
# surfaces


def _scene_on(surface=None, **kw):
    c = Canvas(220, 120, seed=11, surface=surface, **kw)
    c.stroke([(10, 40), (210, 40)], paint="#2244aa", amount=10, width=24, seed=1)
    c.stroke(
        [(10, 85), (210, 85)], paint="#aa3322", amount=6, width=24, pressure=0.2, seed=2
    )
    return c


def test_default_surface_is_the_original_canvas():
    a, b = _scene_on(), _scene_on(surfaces.CANVAS)
    assert np.array_equal(a.thickness, b.thickness)
    assert np.array_equal(a.render(), b.render())


def test_legacy_canvas_arguments_map_onto_the_surface():
    c = Canvas(50, 40, ground="#336699", texture=0.3, weave=2.5, seed=1)
    assert (
        c.surface.relief == 0.3
        and c.surface.scale == 2.5
        and c.surface.ground == "#336699"
    )
    c2 = Canvas(50, 40, surface=surfaces.LINEN_FINE, ground="#ffffff")
    assert c2.surface.texture == "linen" and c2.surface.ground == "#ffffff"


def test_every_surface_preset_builds_and_paints():
    for name, surf in surfaces.ALL.items():
        c = _scene_on(surf)
        img = c.render()
        assert np.isfinite(img).all() and c.thickness.max() > 0, name
        assert c.grain.min() >= 0.0 and c.grain.max() <= 1.0, name


def test_textures_have_relief_except_smooth():
    rng = np.random.default_rng(0)
    for kind in ("canvas", "linen", "cotton", "twill", "paper", "wood"):
        hmap = Surface(texture=kind, scale=4).height_map(80, 80, rng)
        assert hmap.std() > 0.1, kind
    assert Surface(texture="smooth").height_map(40, 40, rng).max() == 0


def _painted_length(c, y):
    cols = np.nonzero(c.thickness[y - 4 : y + 4].max(axis=0) > 0.15)[0]
    return cols.max() - cols.min() if len(cols) else 0


def test_absorbent_ground_drinks_paint():
    plain, chalk = (
        Canvas(400, 60, seed=3),
        Canvas(400, 60, seed=3, surface=surfaces.GESSO_PANEL),
    )
    for c in (plain, chalk):
        c.stroke([(10, 30), (390, 30)], paint="#2244aa", amount=8, width=24, seed=1)
    assert _painted_length(chalk, 30) < 0.9 * _painted_length(plain, 30)


def test_rough_tooth_breaks_up_a_light_stroke():
    def coverage(surf):
        c = Canvas(300, 80, seed=4, surface=surf)
        c.stroke(
            [(10, 40), (200, 40)],
            paint="#2244aa",
            amount=12,
            width=28,
            pressure=0.2,
            seed=1,
        )
        return float((c.thickness[30:50, 30:170] > 0.05).mean())

    assert coverage(surfaces.LINEN_COARSE) < 0.85 * coverage(surfaces.COPPER)
    assert coverage(surfaces.COPPER) > 0.95


def test_slick_surface_wipes_clean_absorbent_one_stains():
    def left_after_wiping(surf):
        c = Canvas(240, 80, seed=5, surface=surf)
        c.stroke([(10, 40), (230, 40)], paint="#aa3322", amount=20, width=30, seed=1)
        before = c.thickness[30:50, 40:200].sum()
        for k in range(2):  # wipe back with a rag, twice
            c.stroke(
                [(230, 40), (10, 40)],
                brush=brushes.RAG,
                width=34,
                pressure=0.6,
                seed=2 + k,
            )
        return c.thickness[30:50, 40:200].sum() / before

    assert left_after_wiping(surfaces.COPPER) < 0.6 * left_after_wiping(
        surfaces.GESSO_PANEL
    )


def test_paint_sinks_in_on_absorbent_grounds():
    def seen(surf):
        c = Canvas(200, 80, seed=6, surface=surf)
        c.stroke([(10, 40), (190, 40)], paint="#2a211b", amount=30, width=34, seed=1)
        return c.render(lighting=False)[30:50, 60:120].mean()

    assert seen(surfaces.GESSO_PANEL) > seen(surfaces.LINEN_FINE) + 0.02


def test_toned_ground():
    white = Canvas(40, 40, seed=7, surface=surfaces.COTTON_DUCK, texture=0)
    toned = Canvas(
        40, 40, seed=7, surface=surfaces.COTTON_DUCK.with_(toning="#a0522d"), texture=0
    )
    r, g, b = toned.render(lighting=False).mean((0, 1))
    assert (
        r > g > b
        and toned.render(lighting=False).mean() < white.render(lighting=False).mean()
    )


def test_surface_validation_and_customisation():
    assert surfaces.LINEN_FINE.with_(relief=2.0).relief == 2.0
    assert surfaces.LINEN_FINE.relief == 0.9  # presets are immutable
    bad_surfaces: list[dict[str, Any]] = [
        {"texture": "velvet"},
        {"scale": 0},
        {"absorbency": 1.5},
        {"tooth": -0.1},
    ]
    for bad in bad_surfaces:
        with pytest.raises(ValueError):
            Surface(**bad)


def test_canvas_with_surface_pickles():
    import pickle

    c = pickle.loads(pickle.dumps(Canvas(60, 40, seed=8, surface=surfaces.GESSO_PANEL)))
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    assert c.surface.name == "Gesso panel" and np.isfinite(c.render()).all()


def test_rag_only_lifts_paint():
    c = Canvas(200, 80, seed=12)
    c.stroke([(10, 40), (190, 40)], paint="#aa3322", amount=20, width=30, seed=1)
    before = c.thickness.sum()
    rag = c.stroke(
        [(10, 40), (190, 40)], brush=brushes.RAG, paint="#00ff00", amount=5, width=30
    )
    assert c.thickness.sum() < 0.7 * before  # paint was lifted ...
    # ... into the rag, and none laid down
    assert rag.volume > 0


# --------------------------------------------------------------------------
# anti-aliasing


def test_stroke_edges_are_anti_aliased():
    # a 1 px diagonal line: most of its pixels are only partly covered
    c = Canvas(120, 120, seed=13, texture=0)
    c.stroke([(10, 110), (100, 15)], paint="#000000", amount=160, width=1.0)
    img = c.render(lighting=False).mean(-1)
    ground = img[5, 115]
    painted = img < ground - 0.02
    assert (painted & (img > 0.2)).sum() > 0.5 * painted.sum()
    # a wide stroke at a slant: every column crossing its edge has an in-between pixel
    c = Canvas(120, 120, seed=13, texture=0)
    c.stroke([(5, 30), (115, 70)], paint="#000000", amount=60, width=16)
    img = c.render(lighting=False).mean(-1)
    lo, hi = img.min() + 0.05, ground - 0.05
    cols = [x for x in range(20, 100) if ((img[:, x] > lo) & (img[:, x] < hi)).any()]
    assert len(cols) > 0.9 * 80
    # coverage stays within the pixel, and drying bakes it into the ground
    assert c.top_a.max() <= 1.0 and c.top_a.min() >= 0.0
    before = c.render(lighting=False)
    c.dry()
    assert c.top_a.max() == 0 and np.abs(c.render(lighting=False) - before).max() < 2e-3


def test_old_canvas_pickles_without_coverage_still_render():
    import pickle

    c = Canvas(60, 40, seed=14)
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    state = c.__getstate__()
    del state["top_a"], state["und_a"]
    old = Canvas.__new__(Canvas)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    assert np.isfinite(old.render()).all() and old.render(lighting=False).std() > 0.01


# --------------------------------------------------------------------------
# loading several paints on one brush


def _rgb_rows(c, x0, x1):
    return c.render(lighting=False)[:, x0:x1].mean(axis=1)


def test_side_loading_puts_each_paint_on_its_side():
    c = Canvas(300, 100, seed=21, texture=0)
    brush = (
        Brush(brushes.FLAT_SABLE.sized(40))
        .load("#c8201e", 10)
        .load("#f6f5f0", 8, across="left")
    )
    # moving right: the brush's left is up
    c.stroke([(20, 50), (280, 50)], brush=brush)
    rows = _rgb_rows(c, 60, 200)
    upper, lower = rows[36:46].mean(axis=0), rows[54:64].mean(axis=0)
    # the white side is the lighter one
    assert upper.sum() > lower.sum() + 0.3
    assert lower[0] > lower[1] + 0.3  # the other side stays red


def test_turning_a_flat_past_90_degrees_swaps_its_sides():
    def lighter_side(angle):
        c = Canvas(300, 100, seed=22, texture=0)
        brush = (
            Brush(brushes.FLAT_SABLE.sized(40))
            .load("#c8201e", 10)
            .load("#f6f5f0", 8, across="left")
        )
        c.stroke([(20, 50), (280, 50)], brush=brush, angle=angle)
        rows = _rgb_rows(c, 60, 200).sum(axis=1)
        return "up" if rows[36:46].mean() > rows[54:64].mean() else "down"

    assert lighter_side(0) == "up" and lighter_side(180) == "down"


def test_tip_loading_comes_off_first():
    c = Canvas(400, 80, seed=23, texture=0)
    brush = (
        Brush(brushes.ROUND.sized(24))
        .load("#4f6b25", 12)
        .load("#f7c600", 1.0, depth="tip")
    )
    c.stroke([(20, 40), (380, 40)], brush=brush)
    img = c.render(lighting=False)
    start, later = (
        img[36:44, 22:32].mean(axis=(0, 1)),
        img[36:44, 200:260].mean(axis=(0, 1)),
    )
    assert start[0] > later[0] + 0.12  # yellow first, then green


def test_partly_mixed_loads_are_marbled():
    def streakiness(mixing):
        c = Canvas(300, 100, seed=24, texture=0)
        brush = Brush(brushes.FILBERT_SABLE.sized(50)).load_mix(
            [("#c8201e", 1), ("#f7c600", 1)], 12, mixing=mixing, seed=1
        )
        c.stroke([(20, 50), (280, 50)], brush=brush, streaks=0.0)
        # green channel varies with the red/yellow share
        return _rgb_rows(c, 60, 160)[35:65, 1].std()

    assert streakiness(0.1) > 3 * streakiness(1.0)


def test_blending_controls_how_long_two_paints_stay_apart():
    def contrast(blending):
        c = Canvas(700, 100, seed=25, texture=0)
        b = Brush(brushes.FLAT_HOG.sized(40).with_(blending=blending))
        b.load("#c8201e", 30, across="left").load("#f7c600", 30, across="right")
        c.stroke([(20, 50), (680, 50)], brush=b, pressure=0.7)
        rows = _rgb_rows(c, 550, 650)
        return float(np.abs(rows[36:44, 1].mean() - rows[56:64, 1].mean()))

    assert contrast(0.05) > 2 * contrast(1.0)


def test_working_the_brush_blends_its_columns():
    b = (
        Brush(brushes.FLAT_SABLE)
        .load("#c8201e", 5, across="left")
        .load("#f7c600", 5, across="right")
    )

    # columns that are neither the red nor the yellow
    def blended_columns(brush):
        g = np.array([int(h[3:5], 16) for h in brush.colors()])
        return int(((g > 50) & (g < 180)).sum())

    before = blended_columns(b)
    v = b.volume
    b.work(4)
    assert blended_columns(b) >= before + 4 and abs(b.volume - v) < 1e-6 * v


def test_contamination_stays_on_the_side_that_touched():
    c = Canvas(300, 120, seed=26)
    c.stroke(
        [(140, 75), (160, 75)],
        "#c8201e",
        amount=20,
        brush=brushes.FLAT_HOG.sized(30),
        angle=90,
    )
    b = Brush(brushes.FLAT_HOG.sized(40)).load("#f6f5f0", 14)
    # only its lower (right) side crosses the red
    c.stroke([(20, 60), (280, 60)], brush=b)
    reds = [int(h[1:3], 16) - int(h[5:7], 16) for h in b.colors()]
    assert np.mean(reds[-6:]) > np.mean(reds[:6]) + 3


def test_multi_load_conserves_paint():
    c = Canvas(300, 120, seed=27)
    b = Brush(brushes.FLAT_HOG.sized(30))
    b.load("#c8201e", 6).load("#f6f5f0", 3, across="left").load(
        "#f7c600", 1, depth="tip"
    )
    b.load_mix([("#1b3fa0", 1), ("#f6f5f0", 2)], 4, mixing=0.3, across="right", seed=2)
    loaded = b.volume
    c.stroke([(20, 60), (150, 40), (280, 70)], brush=b, angle=lambda u: 120 * u)
    assert c.thickness.sum() + b.volume == pytest.approx(loaded, rel=1e-3)


def test_stroke_accepts_a_list_of_paints():
    c = Canvas(200, 80, seed=28, texture=0)
    c.stroke(
        [(10, 40), (190, 40)],
        paint=[("#c8201e", 8), ("#f6f5f0", 4, "left"), ("#f7c600", 0.5, None, "tip")],
        brush=brushes.FLAT_SABLE.sized(30),
    )
    assert c.thickness.sum() > 0


def test_across_options_and_errors():
    from oilpaint.brush import _across_weights

    for across in (
        None,
        "all",
        "left",
        "right",
        "center",
        "edges",
        ("left", 0.3),
        ("right", 0.2),
        (0.2, 0.6),
        lambda x: x,
        np.ones(COLUMNS),
    ):
        w = _across_weights(across)
        assert w.shape == (COLUMNS,) and abs(w.sum() - 1) < 1e-9 and w.min() >= 0
    assert (
        _across_weights("left")[:4].sum() > 0.1
        and _across_weights("left")[-4:].sum() < 1e-6
    )
    for bad in ("top", ("up", 0.5), (2.0, 3.0), [1, 2, 3]):
        with pytest.raises(ValueError):
            _across_weights(bad)
    with pytest.raises(ValueError):
        Brush().load("#ffffff", 1, depth="handle")


# --------------------------------------------------------------------------
# rolling a round, and loading at different depths


def _halves(c, x0, x1, y=50, h=8):
    img = c.render(lighting=False)
    up = img[y - 2 - h : y - 2, x0:x1].mean(axis=(0, 1))
    down = img[y + 3 : y + 3 + h, x0:x1].mean(axis=(0, 1))
    return up, down


def test_side_loading_a_round_puts_each_paint_on_its_side():
    c = Canvas(300, 100, seed=31, texture=0)
    b = (
        Brush(brushes.ROUND.sized(36))
        .load("#c8201e", 10, across="left")
        .load("#f7c600", 10, across="right")
    )
    c.stroke([(20, 50), (280, 50)], brush=b, roll=0)
    up, down = _halves(c, 40, 120)
    # red (low green) on the brush's left = up, yellow down
    assert up[1] < down[1] - 0.2


def test_rolling_a_round_turns_the_other_side_to_the_paper():
    def ends(roll):
        c = Canvas(500, 100, seed=32, texture=0)
        b = Brush(brushes.ROUND.sized(36).with_(blending=0.05))
        b.load("#c8201e", 16, across="left").load("#f7c600", 16, across="right")
        c.stroke([(20, 50), (480, 50)], brush=b, roll=roll)
        return _halves(c, 40, 80), _halves(c, 400, 440)

    # half a turn over the stroke
    (s_up, s_down), (e_up, e_down) = ends(lambda u: 180 * u)
    # starts red on top
    assert s_up[1] < s_down[1] - 0.15
    # ends yellow on top
    assert e_up[1] > e_down[1] + 0.15
    (s_up, s_down), (e_up, e_down) = ends(0)
    # without rolling it stays red on top
    assert e_up[1] < e_down[1] - 0.15


def test_dip_depth_spreads_the_load_over_tip_belly_and_heel():
    from oilpaint.brush import _depth_weights

    assert np.allclose(_depth_weights(0.1), [1, 0, 0])
    w = _depth_weights(1.0)
    assert np.all(w > 0) and abs(w.sum() - 1) < 1e-12 and w[1] > w[0]
    b = Brush(brushes.FLAT_HOG).load("#c8201e", 6, depth=0.5)
    assert (
        b.heel[:, -1].sum() == 0
        and b.tip[:, -1].sum() > 0
        and b.reservoir[:, -1].sum() > 0
    )
    for bad in (0, 1.5, "ferrule"):
        with pytest.raises(ValueError):
            _depth_weights(bad)


def test_paint_in_the_heel_seeps_out_slowly():
    def stroke(depth):
        c = Canvas(900, 80, seed=33, texture=0)
        b = Brush(brushes.FLAT_HOG.sized(30)).load("#1b3fa0", 8, depth=depth)
        v0 = b.volume
        c.stroke([(20, 40), (880, 40)], brush=b)
        return c.thickness[30:50].mean(axis=0), b.volume / v0, b.heel[:, -1].sum() / v0

    belly, belly_left, _ = stroke("belly")
    _deep, deep_left, deep_heel = stroke(1.0)
    heel, heel_left, _ = stroke("heel")
    assert belly_left < 0.1  # a normal load is used up
    # dipped to the ferrule: paint stays in the brush
    assert deep_left > 0.2 and deep_heel > 0.05
    # heel paint only comes out faintly
    assert 0 < heel[40:120].mean() < 0.5 * belly[40:120].mean()
    assert heel_left > 0.5


def test_roll_and_depth_conserve_paint():
    c = Canvas(400, 120, seed=34)
    b = Brush(brushes.ROUND.sized(30)).load("#c8201e", 5, across="left", depth=0.8)
    b.load("#f6f5f0", 4, across="right", depth="heel").load("#f7c600", 1, depth="tip")
    loaded = b.volume
    c.stroke([(20, 60), (200, 30), (380, 80)], brush=b, roll=lambda u: 540 * u)
    assert c.thickness.sum() + b.volume == pytest.approx(loaded, rel=1e-3)


def test_pressing_hard_squeezes_paint_out_of_the_heel():
    def second_stroke(pressure):
        b = Brush(brushes.FLAT_HOG.sized(30)).load("#1b3fa0", 8, depth=1.0)
        # uses up tip and belly
        Canvas(600, 80, seed=35).stroke([(20, 40), (580, 40)], brush=b)
        c = Canvas(600, 80, seed=36)
        c.stroke([(20, 40), (580, 40)], brush=b, pressure=pressure)
        return c.thickness.sum()

    assert second_stroke(0.9) > 2 * second_stroke(0.4)


def test_no_stray_paint_after_a_brush_runs_dry():
    for depth in ("belly", 1.0):
        for seed in (1, 6, 9):
            c = Canvas(1000, 80, seed=seed)
            b = Brush(brushes.FLAT_HOG.sized(30)).load("#1b3fa0", 8, depth=depth)
            c.stroke([(30, 40), (970, 40)], brush=b)
            cols = np.nonzero((c.thickness[20:60] > 0.02).any(axis=0))[0]
            assert (cols.max() - 30) / 30 < 10, (depth, seed)
    # paint up in the heel: only a little dry-brush scratching comes out
    for seed in (1, 6, 9):
        c = Canvas(1000, 80, seed=seed)
        b = Brush(brushes.FLAT_HOG.sized(30)).load("#1b3fa0", 8, depth="heel")
        loaded = b.volume
        c.stroke([(30, 40), (970, 40)], brush=b)
        assert c.thickness.sum() < 0.15 * loaded and b.volume > 0.8 * loaded


@pytest.mark.parametrize("shape", ["round", "flat"])
def test_thin_brushes_go_as_far_as_wide_ones(shape):
    # `amount` is in brush widths, whatever the size: a 3 px liner and a 30 px
    # brush loaded with 6 both lay about 6 widths of solid stroke
    reach = []
    for width in (3, 6, 30):
        length = int(width * 30) + 60
        c = Canvas(length, 80, seed=0)
        c.stroke(
            [(20, 40), (length - 20, 40)],
            paint="#1b3fa0",
            amount=6,
            brush=brushes.make(shape, width, "synthetic"),
            seed=0,
        )
        th = c.thickness.sum(axis=0)
        ref = th[20 + width : 20 + 3 * width].mean()
        reach.append((np.nonzero(th > 0.5 * ref)[0].max() - 20) / width)
    assert all(4.5 < r < 7.5 for r in reach), reach
    assert max(reach) - min(reach) < 1.2, reach


# --------------------------------------------------------------------------
# impasto: paint body, relief, ridges and knives


def test_paint_body_mixes_by_volume():
    assert P.CADMIUM_RED.body == pytest.approx(0.7)
    thinned = Paint.mix((P.CADMIUM_RED, 1), (P.MEDIUM, 1))
    stiffened = Paint.mix((P.CADMIUM_RED, 2), (P.IMPASTO_MEDIUM, 1))
    assert thinned.body < 0.4 < 0.75 < stiffened.body
    assert (
        Paint("#123456", body=0.2).body == 0.2
        and P.CADMIUM_RED.with_body(0.9).body == 0.9
    )
    with pytest.raises(ValueError):
        Paint("#123456", body=1.5)
    from oilpaint import Brush

    loaded = Brush(brushes.FLAT_HOG).load(thinned, 5).paint
    assert loaded is not None
    assert loaded.body == pytest.approx(thinned.body, abs=1e-6)


def _grooves(c, rows, cols):
    t = c.thickness[rows, cols]
    return float(np.abs(np.diff(t, axis=0)).mean())


def test_a_heavy_load_builds_relief():
    thin, thick = Canvas(300, 100, seed=1), Canvas(300, 100, seed=1)
    for c, amount in ((thin, 6), (thick, 40)):
        c.stroke(
            [(30, 50), (270, 50)],
            paint=P.CADMIUM_RED,
            amount=amount,
            brush=brushes.FLAT_HOG.sized(36),
            seed=1,
        )
    core = (slice(38, 62), slice(60, 200))
    assert thick.thickness[core].mean() > 2.5 * thin.thickness[core].mean()
    # deep bristle grooves
    assert _grooves(thick, *core) > 3 * _grooves(thin, *core)


def test_pressing_an_overloaded_brush_squeezes_ridges_out_at_the_edges():
    def outside(pressure):
        c = Canvas(300, 100, seed=1)
        c.stroke(
            [(30, 50), (270, 50)],
            paint=P.CADMIUM_RED,
            amount=45,
            brush=brushes.FLAT_HOG.sized(36),
            pressure=pressure,
            seed=1,
        )
        t = c.thickness[:, 80:220]
        return float(t[:31].sum() + t[69:].sum())  # beyond the brush's 36 px

    assert outside(0.9) > 3 * outside(0.2)


def test_fluid_paint_levels_out_stiff_paint_holds_its_marks():
    def grooves(paint):
        c = Canvas(400, 100, seed=1)
        c.stroke(
            [(30, 50), (370, 48)],
            paint=paint,
            amount=40,
            brush=brushes.FLAT_HOG.sized(36),
            seed=61,
        )
        return _grooves(c, slice(38, 62), slice(100, 300))

    tube = grooves(P.CADMIUM_RED)
    fluid = grooves(Paint.mix((P.CADMIUM_RED, 1), (P.MEDIUM, 3)))
    stiff = grooves(Paint.mix((P.CADMIUM_RED, 2), (P.IMPASTO_MEDIUM, 1)))
    assert fluid < 0.4 * tube and stiff >= 0.95 * tube


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
    from oilpaint import Brush

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
    # scraped down
    assert after[55:66, cols].mean() < 0.5 * before[55:66, cols].mean()
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
        loaded += amount * 0.9 * br.width**2 - br.volume
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


def test_old_canvas_pickles_without_paint_body_still_render():
    import pickle

    c = Canvas(60, 40, seed=14)
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    state = c.__getstate__()
    state["top"], state["und"] = (
        state["top"][..., :-1].copy(),
        state["und"][..., :-1].copy(),
    )
    for name in ("_floor", "_hollows", "_peaks_fine", "_peaks_wide"):
        del state[name]
    old = Canvas.__new__(Canvas)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    assert old.top.shape == c.top.shape and np.isfinite(old.render()).all()
    old.stroke([(5, 30), (55, 30)], paint="#993366", amount=5, width=12)


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


def _wet_knife_field(n=6, width=600):
    purple = Paint.mix(
        (P.DIOXAZINE_PURPLE, 2), (P.QUINACRIDONE_MAGENTA, 1), (P.TITANIUM_WHITE, 1)
    )
    c = Canvas(width, 200, seed=4)
    for k in range(n):
        y = 30 + 30 * k
        c.stroke(
            [(20, y), (width - 20, y + 3)],
            paint=purple,
            amount=200,
            brush=brushes.KNIFE_STRAIGHT.sized(60),
            pressure=0.35,
            seed=k + 1,
        )
    return c


def _ripple_along_x(surface, rows, cols):
    band = surface[rows, cols].mean(axis=0)
    return float(
        (band - np.convolve(band, np.ones(21) / 21, mode="same"))[15:-15].std()
    )


def test_overlapping_wet_knife_slabs_are_smooth():
    # a knife laid over the edge of a wet slab doesn't leave a bead at every step
    c = _wet_knife_field(n=2)
    surface = (c._floor + c.top_v + c.und_v)[86:100, 120:480]
    ripple = surface - np.apply_along_axis(
        lambda r: np.convolve(r, np.ones(15) / 15, mode="same"), 1, surface
    )
    assert ripple[:, 10:-10].std() < 0.12


def test_a_clean_knife_drag_leaves_no_ribs():
    c = _wet_knife_field()
    c.stroke(
        [(100, 100), (300, 103), (500, 100)],
        brush=brushes.PALETTE_KNIFE.sized(60),
        pressure=0.6,
        seed=8,
    )
    surface = c._floor + c.top_v + c.und_v
    assert _ripple_along_x(surface, slice(90, 112), slice(150, 450)) < 0.3


def test_moved_paint_is_fully_covered():
    # paint dragged or merged into a pixel must not leave it looking part-covered (light
    # specks)
    c = _wet_knife_field(n=3)
    lum = c.render(lighting=False).mean(-1)
    core = lum[50:105, 60:540]
    assert core.max() - np.median(core) < 0.1


def test_colour_bands_wander_on_a_knife_but_not_on_a_brush():
    from oilpaint import Brush

    def boundary_wander(tool):
        c = Canvas(500, 160, seed=3)
        br = (
            Brush(tool)
            .load(P.CADMIUM_RED, 30, across="left")
            .load(P.TITANIUM_WHITE, 30, across="right")
        )
        c.stroke([(40, 80), (460, 80)], brush=br, pressure=0.4, seed=1)
        img = c.render(lighting=False)
        # red minus blue: red side vs white side
        redness = img[40:120, 80:300, 0] - img[40:120, 80:300, 2]
        # width of the red band in each column
        red_rows = (redness > 0.5 * redness.max()).sum(axis=0)
        return float(np.std(red_rows))

    assert boundary_wander(brushes.PALETTE_KNIFE.sized(60)) > 1.0
    assert boundary_wander(brushes.FLAT_SABLE.sized(60)) < 1.0


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


def test_soft_strokes_overlapping_on_a_wet_ground_leave_no_ring():
    # white over white on wet blue: the soft edge of the second stroke must not
    # bury the first one's white in the blue and leave a bluish ring
    c = Canvas(400, 220, seed=5)
    for y in range(20, 220, 40):
        c.stroke(
            [(-30, y), (430, y)],
            paint=P.PHTHALO_BLUE,
            amount=14,
            brush=brushes.FLAT_HOG.sized(60),
            pressure=0.5,
            seed=y,
        )
    soft = brushes.make("round", 60, "squirrel", softness=0.85)
    for k, y in enumerate((95, 125)):
        c.stroke(
            [(60, y), (340, y)],
            brush=Brush(soft).load(P.TITANIUM_WHITE, 12),
            pressure=0.35,
            seed=40 + k,
        )
    img = c.render(lighting=False)
    # down the middle of the strokes
    blueness = (img[..., 2] - img[..., 0])[60:160, 120:280].mean(axis=1)
    between = blueness[95 - 60 : 125 - 60 + 1]
    assert between.max() - max(between[0], between[-1]) < 0.03


def test_palette_match_finds_a_close_mix_of_the_given_paints():
    m = P.match("#c89070", P.OLD_MASTER, over="#5a4532")
    assert m.error < 2.0
    assert all(any(p is q for q in P.OLD_MASTER) for p, _ in m.parts)
    assert len(m.parts) <= 3
    # the mix it returns really looks like that over that ground
    got = sp.srgb_to_lab(m.paint.appearance(1.0, over="#5a4532"))
    assert np.linalg.norm(got - sp.srgb_to_lab(sp.parse_color("#c89070"))) < 2.0


def test_reference_samples_colour_direction_and_detail():
    from oilpaint.reference import Reference

    H, W = 120, 160
    _, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3))
    # a vertical edge: black left, red right
    img[..., 0] = xx > 80
    ref = Reference(img, W, H)
    assert ref.color(120, 60)[0] > 0.9 and ref.color(40, 60)[0] < 0.1
    angle, strength = ref.direction(80, 60, scale=3)
    # strokes run along the edge
    assert abs(abs(np.sin(angle)) - 1.0) < 0.05 and strength > 0.8
    assert ref.detail(80, 60, 4) > ref.detail(20, 60, 4) + 0.1
    path = ref.trace(40, 60, length=60, scale=3)
    assert np.all(path[:, 0] < 80)  # a trace doesn't cross into the red


def test_a_brush_handle_scratches_down_to_the_ground():
    c = Canvas(300, 120, surface=Surface(toning="#6a4a30"), seed=2)
    c.stroke(
        [(10, 60), (290, 60)],
        paint=P.LEAD_WHITE,
        amount=14,
        brush=brushes.FLAT_HOG.sized(60),
        seed=1,
    )
    before, total = (
        float((c.top_v + c.und_v)[60, 150]),
        float((c.top_v + c.und_v).sum()),
    )
    c.stroke(
        [(150, 20), (150, 100)],
        brush=brushes.BRUSH_HANDLE.sized(5),
        pressure=0.9,
        seed=3,
    )
    wet = c.top_v + c.und_v
    assert wet[60, 150] < 0.15 * before  # the groove
    # the paint is pushed aside, not lost
    assert abs(float(wet.sum()) - total) < 0.01 * total
    lum = c.render(lighting=False).mean(-1)
    # the dark ground shows in the scratch
    assert lum[60, 150] < lum[60, 120] - 0.2
