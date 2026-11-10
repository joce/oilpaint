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
