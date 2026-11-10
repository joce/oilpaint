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
