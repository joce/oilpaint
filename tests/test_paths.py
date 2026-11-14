"""Stroke paths."""

import numpy as np

from oilpaint import Bezier, CatmullRom, Polyline


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


def test_a_length_a_whole_number_of_steps_long_is_cut_the_same_whatever_its_last_bit():
    # 80 px in steps of 8: 10 steps in exact arithmetic, and the length a
    # hair either side of 80 after rounding (the end moved by an ulp). The
    # count (and every random number a stroke draws after it) used to
    # follow that hair: 10 steps or 11.
    for path in (CatmullRom, Polyline):
        counts = {
            len(path([(10.0, 20.0), (x1, 20.0)]).sample(8.0)[0])
            for x1 in (90.0, np.nextafter(90.0, 100.0), np.nextafter(90.0, 0.0))
        }
        assert counts == {11}


def test_a_bezier_with_a_whole_even_control_polygon_is_cut_the_same_either_way():
    # Its control polygon is 90 px long, half of it 45 in exact arithmetic
    # and a hair either side of 45 after rounding (the end moved by an ulp):
    # it was cut into 45 + 1 pieces or 44 + 1.
    counts = {
        len(Bezier([(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (x3, 0.0)]).sample(1.0)[0])
        for x3 in (90.0, np.nextafter(90.0, 100.0), np.nextafter(90.0, 0.0))
    }
    assert len(counts) == 1
    dense = {
        len(Bezier([(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (x3, 0.0)])._dense()[0])
        for x3 in (90.0, np.nextafter(90.0, 100.0), np.nextafter(90.0, 0.0))
    }
    assert dense == {47}  # (45 + 1 samples, and the end)
