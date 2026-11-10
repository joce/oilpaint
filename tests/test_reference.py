"""Painting from a reference image."""

import numpy as np


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


def test_an_integer_image_is_on_the_0_255_scale_whatever_its_values():
    from oilpaint.reference import Reference

    # a nearly black image of whole numbers stays nearly black: not read as
    # 0..1 because no value is above 1
    for dtype in (np.uint8, np.int16, np.int64):
        dark = np.ones((20, 30, 3), dtype)
        assert Reference(dark, 30, 20).color(15, 10).max() < 0.01, dtype
    # floats keep their rule: 0..1, unless a value is above 1
    half = Reference(np.full((20, 30, 3), 0.5), 30, 20).color(15, 10)
    assert np.allclose(half, 0.5, atol=0.01)
    wide = Reference(np.full((20, 30, 3), 128.0), 30, 20).color(15, 10)
    assert np.allclose(wide, 128 / 255, atol=0.01)
    # and 0..255 integers as before
    full = Reference(np.full((20, 30, 3), 255, np.int64), 30, 20).color(15, 10)
    assert np.allclose(full, 1.0, atol=0.01)
