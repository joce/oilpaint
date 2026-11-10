"""Canvases pickled, now and by earlier versions."""

import numpy as np

from oilpaint import Canvas, surfaces


def test_canvas_survives_pickling():
    import pickle

    c = Canvas(60, 40, seed=9)
    c = pickle.loads(pickle.dumps(c))
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    assert c.top_v.max() > 0 and c.thickness.max() > 0


def test_canvas_with_surface_pickles():
    import pickle

    c = pickle.loads(pickle.dumps(Canvas(60, 40, seed=8, surface=surfaces.GESSO_PANEL)))
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    assert c.surface.name == "Gesso panel" and np.isfinite(c.render()).all()


def test_old_canvas_pickles_without_coverage_still_render():
    import pickle

    c = Canvas(60, 40, seed=14)
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    state = c.__getstate__()
    del state["top_a"], state["und_a"]
    old = Canvas.__new__(Canvas)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    assert np.isfinite(old.render()).all() and old.render(lighting=False).std() > 0.01


def test_old_canvas_pickles_without_paint_body_still_render():
    import pickle

    c = Canvas(60, 40, seed=14)
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    state = c.__getstate__()
    state["top"], state["und"] = (
        state["top"][..., :-1].copy(),
        state["und"][..., :-1].copy(),
    )
    for name in ("_floor", "_hollows", "_peaks_fine", "_peaks_wide"):
        del state[name]
    old = Canvas.__new__(Canvas)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    assert old.top.shape == c.top.shape and np.isfinite(old.render()).all()
    old.stroke([(5, 30), (55, 30)], paint="#993366", amount=5, width=12)
