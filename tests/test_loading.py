"""Loading a brush: several paints side by side, at different depths, rolled
and worked."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, brushes
from oilpaint.brush import COLUMNS


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
