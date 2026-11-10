"""Impasto: paint body, relief and ridges from brushes."""

import numpy as np
import pytest

from oilpaint import Brush, Canvas, Paint, brushes
from oilpaint import palette as P


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

    loaded = Brush(brushes.FLAT_HOG).load(thinned, 5).paint
    assert loaded is not None
    assert loaded.body == pytest.approx(thinned.body, abs=1e-6)


def _grooves(c, rows, cols):
    t = c.thickness[rows, cols]
    return float(np.abs(np.diff(t, axis=0)).mean())


def test_a_heavy_load_builds_relief():
    ratio, grooves = [], []
    for seed in (1, 2, 3):
        thin, thick = Canvas(300, 100, seed=seed), Canvas(300, 100, seed=seed)
        for c, amount in ((thin, 6), (thick, 40)):
            c.stroke(
                [(30, 50), (270, 50)],
                paint=P.CADMIUM_RED,
                amount=amount,
                brush=brushes.FLAT_HOG.sized(36),
                seed=seed,
            )
        core = (slice(38, 62), slice(60, 200))
        ratio.append(thick.thickness[core].mean() / thin.thickness[core].mean())
        grooves.append(_grooves(thick, *core) / _grooves(thin, *core))
    # (a load of 40 lays about 2.6 x what a load of 6 does)
    assert np.mean(ratio) > 2.3
    # deep bristle grooves
    assert np.mean(grooves) > 3


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
