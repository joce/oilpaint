"""Surfaces: canvas, linen, panels and paper, and what they do to paint."""

from typing import Any

import numpy as np
import pytest

from oilpaint import Canvas, Surface, brushes, surfaces


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


@pytest.mark.parametrize("name", list(surfaces.ALL))
def test_every_surface_preset_builds_and_paints(name):
    c = _scene_on(surfaces.ALL[name])
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
