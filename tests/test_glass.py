"""Reverse painting on glass."""

import numpy as np
import pytest

from oilpaint import Canvas, Paint, surfaces
from oilpaint import palette as P
from oilpaint import spectral as sp


def _on_glass(first, second, reverse, ground=None):
    """``first`` painted on glass and left to dry, then ``second`` across it."""
    glass = surfaces.GLASS if ground is None else surfaces.GLASS.with_(ground=ground)
    c = Canvas(200, 160, surface=glass, reverse=reverse, seed=1)
    c.stroke([(10, 80), (190, 80)], paint=first, amount=10, width=40, seed=1)
    c.dry()
    c.stroke([(100, 5), (100, 155)], paint=second, amount=10, width=34, seed=2)
    return c


def test_reverse_glass_shows_the_first_paint_in_front():
    red, blue = P.CADMIUM_RED, P.ULTRAMARINE_BLUE
    front, back = _on_glass(red, blue, False), _on_glass(red, blue, True)
    r, _g, b = sp.parse_color(front.color_at(100, 80))
    assert b > r  # painted on the front: the blue goes over the red
    r, _g, b = sp.parse_color(back.color_at(100, 80))
    assert r > b + 0.3  # painted on the back: the red, laid first, is in front
    # where they don't overlap, the paint looks the same from either side
    for x, y in ((30, 80), (100, 20)):
        seen = [sp.parse_color(c.color_at(x, y)) for c in (front, back)]
        assert np.abs(seen[0] - seen[1]).max() < 0.02


def test_a_glaze_painted_first_on_reverse_glass_tints_what_goes_behind():
    glaze = Paint.mix((P.ALIZARIN_CRIMSON, 1), (P.MEDIUM, 4))
    white = P.TITANIUM_WHITE
    over = sp.parse_color(_on_glass(glaze, white, False).color_at(100, 80))
    through = sp.parse_color(_on_glass(glaze, white, True).color_at(100, 80))
    assert over[0] - over[1] < 0.05  # on the front the white covers the glaze
    assert through[0] - through[1] > 0.2  # seen through it, the white turns pink


def test_reverse_glass_shows_its_backing_and_looks_the_same_once_dry():
    c = _on_glass(P.CADMIUM_YELLOW, P.PHTHALO_GREEN, True, ground="#141414")
    assert np.abs(sp.parse_color(c.color_at(20, 20)) - 0x14 / 255).max() < 0.01
    before = c.render()
    c.dry()
    assert np.abs(c.render() - before).max() < 1e-4


def test_partly_covered_films_on_reverse_glass_never_reflect_more_than_comes_in():
    from oilpaint.canvas import _film

    rng = np.random.default_rng(1)
    n, N = 5000, sp.N_BANDS
    rf, rb, t2 = np.zeros((n, N)), np.zeros((n, N)), np.ones((n, N))
    for _ in range(3):  # three films on the glass, each partly covering the pixel
        r, tt = _film(
            rng.uniform(0, 8, (n, N)) * rng.uniform(0, 1, (n, 1)) ** 2,
            rng.uniform(0, 15, n),
            rng.uniform(0, 2, n) * rng.uniform(0, 1, n),
            np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(0, 1, n)),
        )
        bounce = 1 / (1 - rb * r)
        rf, rb, t2 = rf + t2 * r * bounce, r + tt * rb * bounce, t2 * tt * bounce**2
    assert (rf + t2 / (1 - rb)).max() <= 1.0  # over a perfectly white backing
    # a white film over half the pixel, in front of a light backing
    r, tt = _film(np.zeros((1, N)), np.full(1, 12.0), np.ones(1), np.full(1, 0.5))
    assert (r + tt * 0.9 / (1 - r * 0.9)).max() < 0.95


def test_reverse_painting_needs_a_clear_support_and_pickles():
    import pickle

    with pytest.raises(ValueError):
        Canvas(40, 40, surface=surfaces.COPPER, reverse=True)
    c = _on_glass(P.CADMIUM_RED, P.ULTRAMARINE_BLUE, True)
    copy = pickle.loads(pickle.dumps(c))
    assert copy.reverse and copy.color_at(100, 80) == c.color_at(100, 80)
