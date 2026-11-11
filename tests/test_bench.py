"""tools/bench.py: the kept state and picture of a scene, and how they compare."""

import importlib.util
import os
import sys
from types import ModuleType

import numpy as np
import pytest

from oilpaint import Canvas, brushes, surfaces
from oilpaint import palette as P


def _load_tool(name: str, path: str) -> ModuleType:
    """Import the script at ``path`` as the module ``name``, as an import
    statement would: registered in ``sys.modules`` before it runs (its
    dataclasses look their module up there), and taken out again if it
    fails. The entries it puts on ``sys.path`` are taken off again."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.path[:] = saved_path
    return module


bench = _load_tool(
    "oilpaint_tools_bench",
    os.path.join(os.path.dirname(__file__), "..", "tools", "bench.py"),
)


def tiny() -> Canvas:
    return Canvas(48, 36, seed=3)


def paint(c, colour=P.CADMIUM_RED):
    c.stroke([(4, 6), (44, 30)], colour, amount=8, width=10)
    c.stroke([(4, 30), (44, 6)], P.TITANIUM_WHITE, brush=brushes.PALETTE_KNIFE)
    return 2, ()


@pytest.fixture
def painted():
    c = tiny()
    paint(c)
    return c


def keep(path, state, picture=None):
    np.savez(path, **state, **(picture or {}))
    return str(path)


def test_picture_holds_the_renders_reflectance_and_height(painted):
    picture = bench.canvas_picture(painted)
    names = {bench.PICTURE + n for n in bench.RENDERS} | {
        bench.PICTURE + "matte",
        bench.PICTURE + "reflectance",
        bench.PICTURE + "height",
    }
    assert set(picture) == names
    assert all(n.startswith(bench.PICTURE) for n in picture)
    assert not any(n.startswith(bench.PICTURE) for n in bench.canvas_state(painted))
    assert picture[bench.PICTURE + "lit"].shape == (36, 48, 3)
    assert picture[bench.PICTURE + "height"].shape == (36, 48)
    # the second lamp and the overhead one light it otherwise
    lit = picture[bench.PICTURE + "lit"]
    assert not np.array_equal(lit, picture[bench.PICTURE + "raking"])
    assert not np.array_equal(lit, picture[bench.PICTURE + "overhead"])
    # ... and the absorbent ground dulls it
    assert not np.array_equal(lit, picture[bench.PICTURE + "matte"])


def test_the_picture_leaves_the_canvas_as_it_was(painted):
    state = bench.canvas_state(painted)  # (references to the canvas's arrays)
    before, surface = bench.digest(state), painted.surface
    bench.canvas_picture(painted)
    assert bench.digest(state) == before
    assert bench.digest(bench.canvas_state(painted)) == before
    assert painted.surface is surface


def test_no_matte_render_behind_glass():
    c = Canvas(40, 30, surface=surfaces.GLASS, reverse=True, seed=3)
    paint(c)
    picture = bench.canvas_picture(c)
    assert bench.PICTURE + "matte" not in picture
    assert bench.PICTURE + "lit" in picture


def test_same_paint_and_same_picture(painted, tmp_path):
    state, picture = bench.canvas_state(painted), bench.canvas_picture(painted)
    path = keep(tmp_path / "s.npz", state, picture)
    again = tiny()
    paint(again)
    assert bench.compare(
        bench.canvas_state(again), path, bench.canvas_picture(again)
    ) == ([], True)


def test_one_bit_of_a_render_is_a_difference(painted, tmp_path):
    state, picture = bench.canvas_state(painted), bench.canvas_picture(painted)
    path = keep(tmp_path / "s.npz", state, picture)
    lit = picture[bench.PICTURE + "lit"].copy()
    lit.view(np.uint32)[5, 7, 1] ^= 1
    picture[bench.PICTURE + "lit"] = lit
    lines, compared = bench.compare(state, path, picture)
    assert compared
    assert len(lines) == 1
    assert lines[0].startswith(f"{bench.PICTURE}lit: 1 of {lit.size} values differ")


def test_a_picture_difference_is_reported_without_a_paint_one(painted, tmp_path):
    state = bench.canvas_state(painted)
    path = keep(tmp_path / "s.npz", state, bench.canvas_picture(painted))
    other = tiny()
    paint(other, P.COBALT_BLUE)
    lines, compared = bench.compare(state, path, bench.canvas_picture(other))
    assert compared
    assert lines
    assert all(line.startswith(bench.PICTURE) for line in lines)
    assert any(line.startswith(bench.PICTURE + "unlit: ") for line in lines)
    assert all("values differ" in line for line in lines)


def test_a_kept_state_without_a_picture_compares_the_paint(painted, tmp_path):
    state = bench.canvas_state(painted)
    path = keep(tmp_path / "old.npz", state)  # as an older bench.py kept it
    picture = bench.canvas_picture(painted)
    assert bench.compare(state, path, picture) == ([], False)
    # ... the paint in full
    other = tiny()
    paint(other, P.COBALT_BLUE)
    lines, compared = bench.compare(
        bench.canvas_state(other), path, bench.canvas_picture(other)
    )
    assert not compared
    assert any(line.startswith("top: ") for line in lines)
    assert not any(bench.PICTURE in line for line in lines)


def test_without_a_picture_the_kept_one_is_not_compared(painted, tmp_path):
    state = bench.canvas_state(painted)
    path = keep(tmp_path / "s.npz", state, bench.canvas_picture(painted))
    assert bench.compare(state, path) == ([], False)


def test_a_picture_array_missing_from_one_side(painted, tmp_path):
    state, picture = bench.canvas_state(painted), bench.canvas_picture(painted)
    path = keep(tmp_path / "s.npz", state, picture)
    del picture[bench.PICTURE + "height"]
    assert bench.compare(state, path, picture) == (
        [f"{bench.PICTURE}height: only in one of the two"],
        True,
    )


def test_verdict():
    v = bench.verdict
    assert v("K", [], [], 3, True) == "same paint and same picture as K"
    assert v("K", [], ["a", "b", "c"], 3, True) == (
        "same paint as K; picture not compared"
    )
    assert v("K", [], ["b"], 3, True) == "same paint as K; picture not compared for b"
    assert v("K", [], [], 3, False) == (
        "same paint as K; picture not compared (--no-picture)"
    )
    assert v("K", ["a: top: ..."], [], 3, True) == "not the same as K:"
    assert v("K", ["a: top: ..."], ["a", "b", "c"], 3, True) == (
        "not the same as K; picture not compared:"
    )


def test_main_saves_and_compares_the_picture(tmp_path, monkeypatch, capsys):
    scenes = [bench.Scene("tiny", "two strokes", tiny, paint, reps=2)]
    monkeypatch.setattr(bench, "SCENES", scenes)
    kept = str(tmp_path / "kept")

    assert bench.main(["--save", kept]) == 0
    out = capsys.readouterr().out
    assert "engine and renderer" in out
    assert "picture: rendered in" in out
    with np.load(os.path.join(kept, "tiny.npz")) as f:
        assert any(n.startswith(bench.PICTURE) for n in f.files)

    assert bench.main(["--compare", kept]) == 0
    assert f"same paint and same picture as {kept}" in capsys.readouterr().out

    assert bench.main(["--compare", kept, "--no-picture"]) == 0
    out = capsys.readouterr().out
    assert "picture: rendered" not in out
    assert f"same paint as {kept}; picture not compared (--no-picture)" in out

    # a state kept by an older bench.py: no picture in it
    path = os.path.join(kept, "tiny.npz")
    with np.load(path) as f:
        paint_only = {n: f[n] for n in f.files if not n.startswith(bench.PICTURE)}
    np.savez(path, **paint_only)
    assert bench.main(["--compare", kept]) == 0
    out = capsys.readouterr().out
    assert out.count("holds no picture") == 1
    assert f"same paint as {kept}; picture not compared" in out
    assert "same picture" not in out

    # a plain timing run renders nothing
    assert bench.main([]) == 0
    out = capsys.readouterr().out
    assert "picture" not in out
    assert "(engine:" in out


def test_main_reports_a_picture_difference(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "SCENES", [bench.Scene("tiny", "", tiny, paint, 1)])
    kept = str(tmp_path / "kept")
    assert bench.main(["--save", kept]) == 0
    path = os.path.join(kept, "tiny.npz")
    with np.load(path) as f:
        arrays = {n: f[n] for n in f.files}
    arrays[bench.PICTURE + "height"] = arrays[bench.PICTURE + "height"] + 1.0
    np.savez(path, **arrays)
    capsys.readouterr()
    assert bench.main(["--compare", kept]) == 1
    out = capsys.readouterr().out
    assert f"not the same as {kept}:" in out
    assert f"tiny: {bench.PICTURE}height: {36 * 48} of {36 * 48} values differ" in out


def test_save_without_the_picture_keeps_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "SCENES", [bench.Scene("tiny", "", tiny, paint, 1)])
    kept = str(tmp_path / "kept")
    assert bench.main(["--save", kept, "--no-picture"]) == 0
    with np.load(os.path.join(kept, "tiny.npz")) as f:
        assert f.files
        assert not any(n.startswith(bench.PICTURE) for n in f.files)
    assert "picture" not in capsys.readouterr().out


def test_a_scene_kept_without_a_picture_is_named(tmp_path, monkeypatch, capsys):
    scenes = [
        bench.Scene("one", "", tiny, paint, 1),
        bench.Scene("two", "", tiny, lambda c: paint(c, P.COBALT_BLUE), 1),
    ]
    monkeypatch.setattr(bench, "SCENES", scenes)
    kept = str(tmp_path / "kept")
    assert bench.main(["--save", kept]) == 0
    path = os.path.join(kept, "two.npz")
    with np.load(path) as f:
        paint_only = {n: f[n] for n in f.files if not n.startswith(bench.PICTURE)}
    np.savez(path, **paint_only)
    capsys.readouterr()
    assert bench.main(["--compare", kept]) == 0
    out = capsys.readouterr().out
    assert out.count("holds no picture") == 1
    assert "so the picture was not compared: two\n" in out
    assert f"same paint as {kept}; picture not compared for two\n" in out


def test_a_selection_of_no_scene_is_refused(tmp_path, capsys):
    for extra in ([], ["--save", str(tmp_path / "s")], ["--compare", "nowhere"]):
        with pytest.raises(SystemExit) as stop:
            bench.main(["-k", "no such scene", *extra])
        assert stop.value.code == 2
        assert "no such scene" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "s")


def test_saving_and_comparing_in_one_directory_is_refused(tmp_path, capsys):
    kept = tmp_path / "kept"
    with pytest.raises(SystemExit) as stop:
        bench.main(["--save", str(kept), "--compare", str(tmp_path / "." / "kept")])
    assert stop.value.code == 2
    assert "same directory" in capsys.readouterr().err
