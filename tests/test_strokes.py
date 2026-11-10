"""Strokes on the canvas: where paint goes, how much, and how it mixes with
the wet paint there."""

import numpy as np
import pytest

from oilpaint import Bezier, Brush, Canvas, Paint, Stroke, brushes
from oilpaint import palette as P
from oilpaint import spectral as sp
from oilpaint.brush import _VOLUME_PER_AMOUNT


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
        loaded += amount * _VOLUME_PER_AMOUNT * b.width**2
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


def _yellow_brush_through_blue(thickness, pressure):
    """A full yellow flat dragged through an even film of wet blue,
    ``thickness`` times as thick as a light stroke lays it."""
    c = Canvas(260, 100, seed=3)
    for y in range(26, 80, 8):
        c.stroke(
            [(10, y), (250, y)],
            paint=P.ULTRAMARINE_BLUE,
            amount=8,
            brush=brushes.FLAT_HOG,
            pressure=0.1,
            seed=1,
        )
    c.top *= thickness
    c.und *= thickness
    brush = Brush(brushes.FLAT_HOG).load(P.CADMIUM_YELLOW, 30)
    c.stroke([(40, 52), (200, 52)], brush=brush, pressure=pressure, seed=2)
    return brush


def _red(brush, depth):
    """Mean red of the paint in one layer of the brush: the yellow loses red as
    blue gets into it."""
    return np.mean([sp.parse_color(h)[0] for h in brush.colors(depth)])


def test_deep_wet_paint_gets_further_into_the_brush():
    thin = _yellow_brush_through_blue(0.3, 0.6)
    thick = _yellow_brush_through_blue(2.0, 0.6)
    # A thin film only dirties the tips ...
    assert _red(thin, "tip") < _red(thin, "belly") - 0.02
    # ... thick paint comes up between the hairs, into the body of the brush.
    assert _red(thick, "belly") < _red(thin, "belly") - 0.06
    # Pressing bends the hairs into the paint: more of it gets in.
    light = _yellow_brush_through_blue(1.0, 0.2)
    hard = _yellow_brush_through_blue(1.0, 0.9)
    assert _red(hard, "belly") < _red(light, "belly") - 0.05


def test_brush_dirty_from_thick_paint_tints_its_whole_next_stroke():
    def next_stroke(brush):
        c = Canvas(360, 80, seed=4)
        c.stroke([(20, 40), (340, 40)], brush=brush, pressure=0.6, seed=3)
        # six brush widths along, well past what the tips held
        return sp.parse_color(c.color_at(180, 40))

    dirty = next_stroke(_yellow_brush_through_blue(2.0, 0.6))
    clean = next_stroke(Brush(brushes.FLAT_HOG).load(P.CADMIUM_YELLOW, 30))
    assert dirty[0] < clean[0] - 0.08  # still greener than the clean yellow


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
