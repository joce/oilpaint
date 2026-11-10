"""Brush types: the shapes, sizes and hair of the tools, and what each paints like."""

import numpy as np
import pytest

from oilpaint import Brush, BrushType, Canvas, Surface, brushes
from oilpaint import palette as P


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


@pytest.mark.parametrize(
    "name", [name for name, b in brushes.ALL.items() if b.deposits]
)
def test_every_preset_paints(name):
    b = brushes.ALL[name]
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


def test_brush_type_validation_for_the_new_fields():
    with pytest.raises(ValueError):
        BrushType(tips=-1)


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
