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
    for name in ("_floor", "_tooth", "_hollows", "_peaks_fine", "_peaks_wide"):
        del state[name]  # (maps of the surface added in later versions)
    old = Canvas.__new__(Canvas)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    assert old.top.shape == c.top.shape and np.isfinite(old.render()).all()
    old.stroke([(5, 30), (55, 30)], paint="#993366", amount=5, width=12)


def test_a_canvas_pickles_its_relief_maps_as_attributes_of_their_own():
    # (the layout every version has pickled: no class private to oilpaint in
    # the pickle, and other versions load it)
    import pickle

    c = Canvas(60, 40, seed=15, surface=surfaces.LINEN_COARSE)
    c.stroke([(5, 20), (55, 20)], paint="#336699", amount=5, width=12)
    c.dry()
    state = c.__getstate__()
    maps = ("_floor", "_tooth", "_hollows", "_peaks_fine", "_peaks_wide")
    assert set(maps) <= set(state) and "_relief_maps" not in state
    assert state is not c.__dict__ and "_relief_maps" in c.__dict__
    assert b"_Relief" not in pickle.dumps(c)
    for name in maps:
        state[name] = state[name] + 1.0  # (kept as they were, not made again)
    old = Canvas.__new__(Canvas)
    old.__setstate__(pickle.loads(pickle.dumps(state)))
    for name in c._relief()._fields:
        kept, made = getattr(old._relief(), name), getattr(c._relief(), name)
        assert np.array_equal(kept, made + 1) and not hasattr(old, "_" + name)
    assert np.array_equal(old.top_v, c.top_v) and np.shares_memory(old.top_v, old.top)
    again = pickle.loads(pickle.dumps(c))
    for name in c._relief()._fields:
        assert np.array_equal(
            getattr(again._relief(), name), getattr(c._relief(), name)
        )
