import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Bezier, Brush, Canvas, CatmullRom, Paint, Polyline, Stroke, mix_colors  # noqa: E402
from oilpaint import palette as P  # noqa: E402
from oilpaint import spectral as sp  # noqa: E402

# --------------------------------------------------------------------------
# spectral / Kubelka-Munk


def test_rgb_reflectance_roundtrip():
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.15, 1.0, (200, 3))
    back = sp.reflectance_to_srgb(sp.linear_rgb_to_reflectance(sp.srgb_to_linear(cols)))
    assert np.abs(back - cols).max() < 1e-3   # < 0.3/255


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
    assert np.allclose(sp.km_layer(K * 200, 400.0, ground), sp.km_opaque(K, 2.0), atol=1e-4)
    assert np.allclose(sp.km_layer(K * 200, 400.0, 0.0), sp.km_opaque(K, 2.0), atol=1e-4)


def test_solve_absorption_reproduces_target():
    target = sp.srgb_to_reflectance("#3a7bd5")
    for S in (0.0, 0.5, 10.0):
        K = sp.solve_absorption(target, S)
        assert np.allclose(sp.km_layer(K, S, 1.0), target, atol=1e-6)


# --------------------------------------------------------------------------
# paints and palette mixing


@pytest.mark.parametrize("color", ["#c8201e", "#1b3fa0", "#f5d000", "#808080", "#ffffff"])
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
    lum = lambda c: float(sp.LUMINANCE @ sp.srgb_to_linear(c))  # noqa: E731
    assert lum(tint.color) > lum(red.color) > lum(shade.color)


def test_glaze_is_transparent_body_color_is_not():
    over_black = "#000000"
    glaze = Paint.mix((P.ALIZARIN_CRIMSON, 1), (P.MEDIUM, 3))
    lum = lambda c: float(sp.LUMINANCE @ sp.srgb_to_linear(c))  # noqa: E731
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
    for path in (CatmullRom([(0, 0), (50, 80), (150, 20)]), Bezier([(0, 0), (40, 100), (120, -40), (160, 60)]),
                 Polyline([(0, 0), (100, 0), (100, 100)])):
        pts, tan, u, _ = path.sample(2.0)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert steps.std() / steps.mean() < 0.1
        assert np.allclose(np.linalg.norm(tan, axis=1), 1.0)
        assert u[0] == 0 and u[-1] == 1


# --------------------------------------------------------------------------
# strokes on the canvas


def _brush_volume(brush):
    return brush.tip[-1] + brush.reservoir[-1]


def test_stroke_puts_paint_along_its_path_only():
    c = Canvas(200, 100, seed=0)
    c.stroke([(20, 50), (100, 45), (180, 50)], paint="#1b3fa0", amount=10, width=20, seed=0)
    t = c.thickness
    assert t[45:52, 30:170].min() > 0.3            # covered along the spine
    assert t[:20].max() == 0 and t[85:].max() == 0  # nothing far away
    assert c.color_at(100, 46) != c.color_at(100, 5)


def test_paint_volume_is_conserved():
    c = Canvas(300, 200, seed=1)
    loaded = 0.0
    for i, (paint, amount) in enumerate([("#1b3fa0", 8), ("#f5d000", 6), ("#c8201e", 5)]):
        b = c.stroke([(40, 40 + 50 * i), (150, 100), (260, 60 + 40 * i)], paint=paint, amount=amount,
                     width=24, pressure=0.7, seed=i)
        loaded += amount * 0.9 * b.width ** 2
        loaded -= _brush_volume(b)            # what stayed on the brush
    blender = c.stroke([(30, 100), (270, 110)], width=30, pressure=0.8)  # clean brush, drags paint
    loaded -= _brush_volume(blender)
    assert c.thickness.sum() == pytest.approx(loaded, rel=1e-3)


def test_pressure_controls_wet_mixing():
    def crossing(pressure):
        c = Canvas(200, 160, seed=2)
        c.stroke([(10, 80), (190, 80)], paint=P.ULTRAMARINE_BLUE, amount=12, width=40, seed=1)
        c.stroke([(100, 5), (100, 155)], paint=P.CADMIUM_YELLOW, amount=12, width=34, pressure=pressure, seed=2)
        return sp.parse_color(c.color_at(100, 80))
    light, hard = crossing(0.1), crossing(1.0)
    # Light touch leaves yellow on top; pressing hard churns in the blue (bluer, darker).
    assert light[2] < hard[2] + 1e-6 or light.sum() > hard.sum()
    assert light[0] > hard[0]


def test_dirty_brush_carries_color():
    c = Canvas(300, 200, seed=3)
    c.stroke([(20, 60), (280, 60)], paint=P.CADMIUM_RED, amount=20, width=40, seed=1)
    brush = c.stroke([(150, 10), (150, 120)], paint=P.TITANIUM_WHITE, amount=6, width=24, seed=2)
    c.stroke([(20, 170), (120, 170)], brush=brush, width=24, seed=3)
    r, g, b = sp.parse_color(c.color_at(40, 170))
    w = P.TITANIUM_WHITE.color
    assert (r - b) > (w[0] - w[2]) + 0.03     # pinkish, not pure white


def test_dry_keeps_colors_and_stops_mixing():
    c = Canvas(160, 120, seed=4)
    c.stroke([(10, 60), (150, 60)], paint=P.ULTRAMARINE_BLUE, amount=10, width=40, seed=1)
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
    c.stroke([(80, 5), (80, 115)], paint=Paint.mix((P.PHTHALO_BLUE, 1), (P.MEDIUM, 4)), amount=10, width=30)
    glazed = sp.parse_color(c.color_at(80, 60))
    assert glazed[1] > glazed[0] and glazed[1] > glazed[2]   # yellow + blue glaze reads green
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
    c.paint(Stroke(path=Bezier([(10, 60), (60, 0), (140, 120), (190, 60)]), paint=P.VIRIDIAN,
                   amount=12, width=[6, 30], pressure=lambda u: 0.2 + 0.6 * u, taper=0.2))
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

from oilpaint import BrushType, brushes  # noqa: E402


def _band_width(canvas, x, thresh=0.15):
    """Number of painted pixels in column x."""
    return int((canvas.thickness[:, x] > thresh).sum())


def _row_width(canvas, y, thresh=0.15):
    return int((canvas.thickness[y, :] > thresh).sum())


def test_default_brush_type_matches_default_stroke():
    a, b = Canvas(160, 80, seed=1), Canvas(160, 80, seed=1)
    a.stroke([(10, 40), (150, 40)], paint="#336699", amount=8, width=20, seed=3)
    b.stroke([(10, 40), (150, 40)], paint="#336699", amount=8, width=20, seed=3, brush=BrushType())
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
        c.stroke([(40, 50), (160, 50)], paint="#aa3322", amount=20, width=30, brush=brush)
        def first_x(y):
            return int(np.argmax(c.thickness[y] > 0.15))
        # how much later the stroke starts near its edge than along its centre
        return first_x(62) - first_x(50)
    assert end_curvature(brushes.FLAT_HOG) <= 2         # square end
    assert end_curvature(BrushType()) >= 4              # rounded end


def test_fixed_angle_calligraphy_thick_and_thin():
    nib = brushes.FLAT_SABLE.with_(angle=0, follow_path=False)      # long side horizontal on the canvas
    c = Canvas(300, 300, seed=4)
    c.stroke([(30, 60), (270, 60)], paint="#111111", amount=30, width=30, brush=nib)       # moving along the nib
    c.stroke([(150, 100), (150, 290)], paint="#111111", amount=30, width=30, brush=nib)    # moving across it
    assert _band_width(c, 100) < 12             # horizontal stroke: thin
    assert _row_width(c, 200) > 24              # vertical stroke: wide


def test_pointed_round_thins_with_light_pressure():
    def width_at(p):
        c = Canvas(200, 100, seed=5)
        c.stroke([(20, 50), (180, 50)], paint="#223355", amount=30, brush=brushes.ROUND_SABLE.sized(20), pressure=p)
        return _band_width(c, 100)
    assert width_at(0.1) < 0.6 * width_at(0.9)


def test_angled_brush_paints_with_its_toe_first():
    c = Canvas(200, 100, seed=6)
    c.stroke([(20, 50), (180, 50)], paint="#223355", amount=30, brush=brushes.DAGGER.sized(40), pressure=0.2)
    col = c.thickness[:, 100]
    above, below = col[30:50].sum(), col[51:71].sum()
    assert max(above, below) > 2.5 * max(min(above, below), 1e-3)


def test_fan_leaves_separate_clumps():
    c = Canvas(200, 120, seed=7)
    c.stroke([(20, 60), (180, 60)], paint="#335522", amount=30, brush=brushes.FAN_HOG.sized(60))
    col = c.thickness[25:95, 100] > 0.1
    runs = np.count_nonzero(np.diff(col.astype(int)) == 1)
    assert runs >= 5


def test_palette_knife_is_smooth():
    c = Canvas(200, 100, seed=8)
    c.stroke([(20, 50), (180, 50)], paint="#dddddd", amount=40, brush=brushes.PALETTE_KNIFE.sized(36))
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
    assert brushes.FLAT_HOG.width == 24             # presets are immutable
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
    assert 20 <= _band_width(c, 100) <= 28      # the flat's own 24 px width


# --------------------------------------------------------------------------
# surfaces

from oilpaint import Surface, surfaces  # noqa: E402


def _scene_on(surface=None, **kw):
    c = Canvas(220, 120, seed=11, surface=surface, **kw)
    c.stroke([(10, 40), (210, 40)], paint="#2244aa", amount=10, width=24, seed=1)
    c.stroke([(10, 85), (210, 85)], paint="#aa3322", amount=6, width=24, pressure=0.2, seed=2)
    return c


def test_default_surface_is_the_original_canvas():
    a, b = _scene_on(), _scene_on(surfaces.CANVAS)
    assert np.array_equal(a.thickness, b.thickness)
    assert np.array_equal(a.render(), b.render())


def test_legacy_canvas_arguments_map_onto_the_surface():
    c = Canvas(50, 40, ground="#336699", texture=0.3, weave=2.5, seed=1)
    assert c.surface.relief == 0.3 and c.surface.scale == 2.5 and c.surface.ground == "#336699"
    c2 = Canvas(50, 40, surface=surfaces.LINEN_FINE, ground="#ffffff")
    assert c2.surface.texture == "linen" and c2.surface.ground == "#ffffff"


def test_every_surface_preset_builds_and_paints():
    for name, surf in surfaces.ALL.items():
        c = _scene_on(surf)
        img = c.render()
        assert np.isfinite(img).all() and c.thickness.max() > 0, name
        assert 0.0 <= c.grain.min() and c.grain.max() <= 1.0, name


def test_textures_have_relief_except_smooth():
    rng = np.random.default_rng(0)
    for kind in ("canvas", "linen", "cotton", "twill", "paper", "wood"):
        hmap = Surface(texture=kind, scale=4).height_map(80, 80, rng)
        assert hmap.std() > 0.1, kind
    assert Surface(texture="smooth").height_map(40, 40, rng).max() == 0


def _painted_length(c, y):
    cols = np.nonzero(c.thickness[y - 4:y + 4].max(axis=0) > 0.15)[0]
    return cols.max() - cols.min() if len(cols) else 0


def test_absorbent_ground_drinks_paint():
    plain, chalk = Canvas(400, 60, seed=3), Canvas(400, 60, seed=3, surface=surfaces.GESSO_PANEL)
    for c in (plain, chalk):
        c.stroke([(10, 30), (390, 30)], paint="#2244aa", amount=8, width=24, seed=1)
    assert _painted_length(chalk, 30) < 0.9 * _painted_length(plain, 30)


def test_rough_tooth_breaks_up_a_light_stroke():
    def coverage(surf):
        c = Canvas(300, 80, seed=4, surface=surf)
        c.stroke([(10, 40), (200, 40)], paint="#2244aa", amount=12, width=28, pressure=0.2, seed=1)
        return float((c.thickness[30:50, 30:170] > 0.05).mean())
    assert coverage(surfaces.LINEN_COARSE) < 0.85 * coverage(surfaces.COPPER)
    assert coverage(surfaces.COPPER) > 0.95


def test_slick_surface_wipes_clean_absorbent_one_stains():
    def left_after_wiping(surf):
        c = Canvas(240, 80, seed=5, surface=surf)
        c.stroke([(10, 40), (230, 40)], paint="#aa3322", amount=20, width=30, seed=1)
        before = c.thickness[30:50, 40:200].sum()
        for k in range(2):                       # wipe back with a rag, twice
            c.stroke([(230, 40), (10, 40)], brush=brushes.RAG, width=34, pressure=0.6, seed=2 + k)
        return c.thickness[30:50, 40:200].sum() / before
    assert left_after_wiping(surfaces.COPPER) < 0.6 * left_after_wiping(surfaces.GESSO_PANEL)


def test_paint_sinks_in_on_absorbent_grounds():
    def seen(surf):
        c = Canvas(200, 80, seed=6, surface=surf)
        c.stroke([(10, 40), (190, 40)], paint="#2a211b", amount=30, width=34, seed=1)
        return c.render(lighting=False)[30:50, 60:120].mean()
    assert seen(surfaces.GESSO_PANEL) > seen(surfaces.LINEN_FINE) + 0.02


def test_toned_ground():
    white = Canvas(40, 40, seed=7, surface=surfaces.COTTON_DUCK, texture=0)
    toned = Canvas(40, 40, seed=7, surface=surfaces.COTTON_DUCK.with_(toning="#a0522d"), texture=0)
    r, g, b = toned.render(lighting=False).mean((0, 1))
    assert r > g > b and toned.render(lighting=False).mean() < white.render(lighting=False).mean()


def test_surface_validation_and_customisation():
    assert surfaces.LINEN_FINE.with_(relief=2.0).relief == 2.0
    assert surfaces.LINEN_FINE.relief == 0.9                 # presets are immutable
    for bad in (dict(texture="velvet"), dict(scale=0), dict(absorbency=1.5), dict(tooth=-0.1)):
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
    rag = c.stroke([(10, 40), (190, 40)], brush=brushes.RAG, paint="#00ff00", amount=5, width=30)
    assert c.thickness.sum() < 0.7 * before                  # paint was lifted ...
    assert rag.reservoir[-1] + rag.tip[-1] > 0               # ... into the rag, and none laid down


# --------------------------------------------------------------------------
# anti-aliasing

def test_stroke_edges_are_anti_aliased():
    # a 1 px diagonal line: most of its pixels are only partly covered
    c = Canvas(120, 120, seed=13, texture=0)
    c.stroke([(10, 110), (100, 15)], paint="#000000", amount=400, width=1.0)
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
