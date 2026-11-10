"""Input handling of the colour, paint, path, palette, reference and surface
helpers: what is accepted, what is refused, and with what message. None of
these need the compiled stroke engine. The sections from "the engine's
guards" on do: the stroke engine's limits, and what strokes, brushes and
canvases accept."""

import copy
import dataclasses
import math
import os
import pickle
import shutil
import subprocess
import sys
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Brush, BrushType, Canvas, brushes, spline, surfaces
from oilpaint import _engine as eng
from oilpaint import brush as bm
from oilpaint import palette as P
from oilpaint import spectral as sp
from oilpaint.paint import Paint, as_paint
from oilpaint.reference import Reference
from oilpaint.surface import Surface

# --------------------------------------------------------------------------
# spectral


def test_spectrum_to_rgb_can_be_rebuilt():
    # The illuminant spectrum and the XYZ white point used to share a name, so
    # the matrix could only be built once, at import.
    np.testing.assert_allclose(
        sp._build_spectrum_to_rgb(sp.WAVELENGTHS), sp.SPECTRUM_TO_RGB, rtol=0, atol=0
    )
    other = sp._build_spectrum_to_rgb(np.linspace(400.0, 700.0, 16))
    assert other.shape == (3, 16)
    np.testing.assert_allclose(other.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "color, expected",
    [
        ("#ff8000", (1.0, 128 / 255, 0.0)),
        ((1, 1, 1), (1.0, 1.0, 1.0)),  # all at most 1: the 0..1 scale
        ((0, 0, 1), (0.0, 0.0, 1.0)),
        ((255, 128, 0), (1.0, 128 / 255, 0.0)),
        (np.array([255, 128, 0], np.uint8), (1.0, 128 / 255, 0.0)),
        ((255.0, 128.0, 0.0), (1.0, 128 / 255, 0.0)),  # whole floats: 0..255
        ((0.5, 0.25, 1.0), (0.5, 0.25, 1.0)),
        ((1.0 + 1e-9, 0.5, 0.5), (1.0, 0.5, 0.5)),  # rounding overshoot
        ((-0.1, 0.5, 1.0), (0.0, 0.5, 1.0)),  # out of range: clipped
        ((-5, 100, 300), (0.0, 100 / 255, 1.0)),
        # floats on the 0..255 scale, as before
        ((123.4, 56.7, 89.1), (123.4 / 255, 56.7 / 255, 89.1 / 255)),
        ((255, 128.0, 0.5), (1.0, 128 / 255, 0.5 / 255)),
        ((2.0, 0.5, 0.0), (2 / 255, 0.5 / 255, 0.0)),
        # numpy uint8 (an image's pixel) is always 0..255; other ints are not
        (np.array([1, 1, 1], np.uint8), (1 / 255, 1 / 255, 1 / 255)),
        ((np.uint8(1), np.uint8(0), np.uint8(1)), (1 / 255, 0.0, 1 / 255)),
        (np.array([1, 1, 1], np.int64), (1.0, 1.0, 1.0)),
        (np.array([1, 0, 1], np.uint16), (1.0, 0.0, 1.0)),
    ],
)
def test_parse_color_scales(color, expected):
    np.testing.assert_allclose(sp.parse_color(color), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "color", [(1.02, 0.5, 0.5), (1.0000011, 0.5, 0.2), (1.99, 0.0, 0.0)]
)
def test_parse_color_refuses_an_ambiguous_scale(color):
    # (1.02, 0.5, 0.5) used to be read on the 0..255 scale: almost black.
    with pytest.raises(ValueError, match=r"0\.\.1 .* 0\.\.255"):
        sp.parse_color(color)


@pytest.mark.parametrize(
    "color, error, message",
    [
        ((np.nan, 0.5, 0.5), ValueError, "finite"),
        ((0.1, 0.2), ValueError, "3 color components"),
        ("red", ValueError, "#rrggbb"),
        (("a", "b", "c"), TypeError, "#rrggbb"),
    ],
)
def test_parse_color_errors(color, error, message):
    with pytest.raises(error, match=message):
        sp.parse_color(color)


# --------------------------------------------------------------------------
# paint


def test_paint_from_a_paint_applies_opacity_and_strength():
    base = Paint("#3050c0", opacity=0.6, strength=1.5, name="blue", body=0.4)
    # opacity: derived again from the colour, keeping the paint's strength
    glaze = Paint(base, opacity=0.2)
    direct = Paint("#3050c0", opacity=0.2, strength=1.5)
    np.testing.assert_array_equal(glaze.K, direct.K)
    assert glaze.S == direct.S
    assert (glaze.name, glaze.body) == ("blue", 0.4)
    # strength: relative to the paint's own
    strong = Paint(base, strength=2)
    np.testing.assert_allclose(strong.K, 2 * base.K, rtol=1e-15)
    np.testing.assert_allclose(strong.S, 2 * base.S, rtol=1e-15)
    both = Paint(base, opacity=0.2, strength=2)
    np.testing.assert_allclose(
        both.K, Paint("#3050c0", opacity=0.2, strength=3).K, rtol=1e-15
    )
    np.testing.assert_allclose(Paint(strong, opacity=0.2).K, both.K, rtol=1e-15)
    # nothing given: an exact copy, as before
    copy = Paint(base)
    np.testing.assert_array_equal(copy.K, base.K)
    assert copy.S == base.S and copy.K is not base.K


def test_paint_strength_of_a_mix_and_opacity_refused():
    mixed = Paint.mix(("#3050c0", 1), ("#ffffff", 2))
    np.testing.assert_allclose(Paint(mixed, strength=3).K, 3 * mixed.K, rtol=1e-15)
    with pytest.raises(ValueError, match="opacity can only be changed"):
        Paint(mixed, opacity=0.5)
    with pytest.raises(ValueError, match="opacity can only be changed"):
        Paint(P.MEDIUM, opacity=0.5)


@pytest.mark.parametrize("strength", [0, -1, float("nan"), float("inf")])
def test_paint_strength_must_be_positive(strength):
    with pytest.raises(ValueError, match="strength must be a positive number"):
        Paint("#3050c0", strength=strength)
    with pytest.raises(ValueError, match="strength must be a positive number"):
        Paint(P.CADMIUM_RED, strength=strength)


def test_paint_opacity_is_clipped_but_must_be_a_number():
    np.testing.assert_array_equal(
        Paint("#3050c0", opacity=1.5).K, Paint("#3050c0", opacity=1.0).K
    )
    with pytest.raises(ValueError, match="opacity must be a number"):
        Paint("#3050c0", opacity=float("nan"))


def test_mix_accepts_bare_colours_and_refuses_negative_parts():
    bare = Paint.mix("#ffffff", "#000000")
    paired = Paint.mix(("#ffffff", 1), ("#000000", 1))
    np.testing.assert_array_equal(bare.K, paired.K)
    assert bare.S == paired.S
    triple = Paint.mix((1.0, 1.0, 1.0), [0, 0, 0])
    np.testing.assert_array_equal(triple.K, paired.K)
    np.testing.assert_array_equal(
        Paint.mix(P.CADMIUM_RED, (P.TITANIUM_WHITE, 0)).K, P.CADMIUM_RED.K
    )
    with pytest.raises(ValueError, match="non-negative"):
        Paint.mix(("#3050c0", 2), ("#ffffff", -1))
    with pytest.raises(TypeError, match=r"\(paint, parts\)"):
        bad: Any = ("#3050c0", "two")
        Paint.mix(bad)
    with pytest.raises(ValueError, match="positive total volume"):
        Paint.mix(("#3050c0", 0))


def test_as_paint_gives_exactly_the_colour_asked_for():
    first = as_paint((0.5001, 0.2, 0.3))
    second = as_paint((0.5019, 0.2, 0.3))  # the same 8-bit hex
    np.testing.assert_array_equal(second.color, (0.5019, 0.2, 0.3))
    np.testing.assert_array_equal(second.K, Paint((0.5019, 0.2, 0.3)).K)
    assert not np.array_equal(first.K, second.K)
    # each call hands out its own paint: changing one leaves the next alone
    a = as_paint("#102030")
    a.K *= 2
    a.name = "changed"
    b = as_paint("#102030")
    np.testing.assert_array_equal(b.K, Paint("#102030").K)
    assert b.name is None
    assert as_paint(P.CADMIUM_RED) is P.CADMIUM_RED


def test_paint_color_is_read_only():
    for paint in (Paint("#102030"), Paint.mix("#102030", "#ffffff")):
        with pytest.raises(ValueError):
            paint.color[0] = 1.0
    colour = np.array([0.1, 0.2, 0.3])
    Paint(colour)
    colour[0] = 0.5  # the caller's array stays theirs


# --------------------------------------------------------------------------
# paths and profiles


@pytest.mark.parametrize("make", [spline.as_path, spline.Polyline, spline.Bezier])
def test_path_needs_points(make):
    with pytest.raises(ValueError, match="at least one point"):
        make([])


@pytest.mark.parametrize("make", [spline.as_path, spline.Polyline])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_path_points_must_be_finite(make, bad):
    with pytest.raises(ValueError, match=r"finite numbers; point 1 is"):
        make([(0, 0), (bad, 3), (10, 10)])


def test_path_points_must_be_pairs():
    with pytest.raises(ValueError, match="pairs, got 3 numbers"):
        spline.as_path([1, 2, 3])
    with pytest.raises(TypeError, match="pairs of numbers"):
        spline.as_path([("a", 1)])


def test_stroke_reports_a_bad_path_and_takes_a_scalar_width():
    canvas = Canvas(50, 40, seed=1)
    with pytest.raises(ValueError, match="at least one point"):
        canvas.stroke([], paint="#c02010")
    with pytest.raises(ValueError, match="finite"):
        canvas.stroke([(5, 5), (np.nan, 30)], paint="#c02010")
    canvas.stroke(
        [(5, 20), (45, 20)], paint="#c02010", width=lambda u: 12 if u < 0.5 else 4
    )
    painted = canvas.thickness > 0.05
    assert painted[:, 14].sum() > 1.5 * painted[:, 36].sum() > 0


def test_single_point_path_is_still_a_dab():
    for path in (spline.as_path([(10, 20)]), spline.Polyline([(10, 20)])):
        pts, _, u, _ = path.sample(2.0)
        np.testing.assert_array_equal(pts, [[10.0, 20.0]])
        np.testing.assert_array_equal(u, [0.0])


def _samples():
    _, _, u, knot = spline.as_path([(0, 0), (100, 0)]).sample(2.0)
    return u, knot


def test_profile_callables_may_take_one_number():
    u, knot = _samples()
    w = spline.evaluate_profile(lambda u: 30 if u < 0.5 else 10, u, knot, 2)
    np.testing.assert_array_equal(w, np.where(u < 0.5, 30.0, 10.0))
    w = spline.evaluate_profile(lambda u: 10 + 5 * math.sin(u), u, knot, 2)
    np.testing.assert_allclose(w, 10 + 5 * np.sin(u), rtol=1e-15)


def test_vectorised_profile_callables_are_called_once():
    u, knot = _samples()
    calls = []

    def width(v):
        calls.append(v)
        return 10 + 5 * np.sin(v)

    np.testing.assert_array_equal(
        spline.evaluate_profile(width, u, knot, 2), 10 + 5 * np.sin(u)
    )
    assert len(calls) == 1 and calls[0] is u


def test_profile_callable_errors_are_not_hidden():
    u, knot = _samples()
    missing: dict = {}

    def broken(v):  # a genuine error: raised as it is, after one call
        return missing["width"]

    with pytest.raises(KeyError):
        spline.evaluate_profile(broken, u, knot, 2)

    def broken_scalar(v):  # fails on the array, then on each number
        return 30 if v < 0.5 else 1 / 0

    with pytest.raises(ZeroDivisionError) as info:
        spline.evaluate_profile(broken_scalar, u, knot, 2)
    assert "with the array of all u" in str(info.value.__notes__)
    assert isinstance(info.value.__cause__, ValueError)


# --------------------------------------------------------------------------
# palette


def test_match_checks_max_paints():
    with pytest.raises(ValueError, match="max_paints must be 1 or more, got 0"):
        P.match("#808080", P.OLD_MASTER, max_paints=0)
    bad: Any = 1.5
    with pytest.raises(TypeError, match="max_paints must be a whole number"):
        P.match("#808080", P.OLD_MASTER, max_paints=bad)
    m = P.match("#808080", P.OLD_MASTER[:3], max_paints=int(np.int64(1)))
    assert len(m.parts) == 1
    whole: Any = 2.0  # a whole number given as a float, as before
    assert len(P.match("#808080", P.OLD_MASTER[:4], max_paints=whole).parts) <= 2
    with pytest.raises(ValueError, match="1 or more"):
        P.match("#808080", P.OLD_MASTER, max_paints=whole - 2)


def test_match_refuses_a_nan_thickness():
    with pytest.raises(ValueError, match="thickness and over must be finite"):
        P.match("#808080", P.OLD_MASTER[:3], thickness=float("nan"))


# --------------------------------------------------------------------------
# surface


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_integer_ground_images_are_0_to_255(dtype):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (4, 5, 3))
    expected = Surface().ground_reflectance(
        4, 5, np.random.default_rng(1), (pixels / 255.0).astype(np.float64)
    )
    got = Surface().ground_reflectance(
        4, 5, np.random.default_rng(1), pixels.astype(dtype)
    )
    np.testing.assert_array_equal(got, expected)


# --------------------------------------------------------------------------
# reference


def test_reference_keeps_a_bounded_number_of_maps():
    rng = np.random.default_rng(2)
    ref = Reference(rng.random((30, 40, 3)), 40, 30)
    xs, ys = rng.random(50) * 39, rng.random(50) * 29
    # 1.936 and 1.9399 share a map (both round to 1.94), but blurred at each
    # they differ: the box radius changes in between
    first = [ref.color(xs, ys, s) for s in (1.936, 3.0)]
    for k in range(60):
        ref.value(xs, ys, 1.0 + 0.1 * k)
        ref.direction(xs, ys, 1.0 + 0.1 * k)
    assert len(ref._cache) == Reference.max_maps
    # made again after being dropped: the same map, at the scale first used
    again = [ref.color(xs, ys, s) for s in (1.9399, 3.0)]
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    ref.max_maps = 2
    ref.detail(xs, ys, 5.0)
    assert len(ref._cache) == 2


def _toned(toning):
    surface = Surface(toning=toning)
    return surface.ground_reflectance(12, 16, np.random.default_rng(4))


def test_toning_paint_is_used_as_it_is():
    # A paint given as the toning is laid as it is; only a bare colour is made
    # into a thin wash (opacity 0.25). A paint with the same optics but no
    # colour to derive from gives the same ground.
    for paint in (P.RAW_UMBER, Paint.mix((P.RAW_UMBER, 1), (P.LEAD_WHITE, 1))):
        same_optics = Paint.from_optics(paint.K, paint.S)
        np.testing.assert_array_equal(_toned(paint), _toned(same_optics))
    np.testing.assert_array_equal(
        _toned("#6b5638"), _toned(Paint("#6b5638", opacity=0.25))
    )
    assert not np.array_equal(_toned("#6b5638"), _toned(P.RAW_UMBER))


# --------------------------------------------------------------------------
# objects pickled by an older version


def _as_old_pickle(obj, drop=(), **replace):
    """``obj`` pickled and read back as an older version would have written
    it: without the attributes in ``drop``, and with ``replace`` set."""
    for name in drop:
        delattr(obj, name)
    for name, value in replace.items():
        setattr(obj, name, value)
    return pickle.loads(pickle.dumps(obj))


def test_old_pickled_paints_still_work():
    old = _as_old_pickle(Paint(P.RAW_UMBER), drop=["_strength"])
    old_mix = _as_old_pickle(
        Paint.mix((P.RAW_UMBER, 1), (P.LEAD_WHITE, 1)), drop=["_strength"]
    )
    np.testing.assert_array_equal(old.K, P.RAW_UMBER.K)
    np.testing.assert_array_equal(Paint(old).K, P.RAW_UMBER.K)
    assert old.with_body(0.5).body == 0.5
    np.testing.assert_array_equal(Paint(old, strength=2).K, 2 * P.RAW_UMBER.K)
    assert Paint(old_mix).hex == old_mix.hex
    with pytest.raises(ValueError, match="older version"):
        Paint(old, opacity=0.5)
    # new pickles keep everything
    new = pickle.loads(pickle.dumps(P.RAW_UMBER))
    np.testing.assert_array_equal(
        Paint(new, opacity=0.2).K, Paint(P.RAW_UMBER, opacity=0.2).K
    )


def test_old_pickled_references_still_work():
    rng = np.random.default_rng(5)
    ref = Reference(rng.random((20, 30, 3)), 30, 20)
    expected = ref.color(10.5, 7.5, 2.0)
    old = _as_old_pickle(ref, drop=["_made_at"], _cache=dict(ref._cache))
    np.testing.assert_array_equal(old.color(10.5, 7.5, 2.0), expected)
    for k in range(Reference.max_maps + 3):
        old.value(3.0, 4.0, 1.0 + 0.5 * k)
    assert len(old._cache) == Reference.max_maps


def test_stronger_mix_has_its_own_colour():
    def mixed():
        return Paint.mix((P.PHTHALO_BLUE, 1), (P.MEDIUM, 4))

    seen = mixed()
    _ = seen.color  # its colour worked out (and kept) before the copy
    for base in (mixed(), seen):
        strong = Paint(base, strength=3)
        np.testing.assert_array_equal(strong.color, strong.appearance())
        assert strong.hex != base.hex
    assert Paint(seen, strength=3).hex == Paint(mixed(), strength=3).hex
    # a paint made from a colour keeps that colour, as Paint(color, strength=)
    assert Paint(P.CADMIUM_RED, strength=2).hex == P.CADMIUM_RED.hex


def test_profile_function_results_are_checked():
    u, knot = _samples()
    n = len(u)
    # a number, one per u, or something that spreads to one per u: as before
    for f in (lambda v: 3, lambda v: np.array([3.0]), lambda v: np.full(n, 3.0)):
        np.testing.assert_array_equal(
            spline.evaluate_profile(f, u, knot, 2), np.full(n, 3.0)
        )

    def mixed_types(v) -> Any:  # a numpy int or a float, by u
        return np.int8(2) if v < 0.5 else 1.0

    np.testing.assert_array_equal(
        spline.evaluate_profile(mixed_types, u, knot, 2),
        np.where(u < 0.5, 2.0, 1.0),
    )
    for f in (
        lambda v: np.atleast_1d(30 if v < 0.5 else 10),  # one array per u
        lambda v: v[:, None] * 10,
        lambda v: np.zeros(n + 1),
    ):
        with pytest.raises(ValueError, match="one number per u"):
            spline.evaluate_profile(f, u, knot, 2)
    not_numbers: list[Any] = [
        lambda v: None,
        lambda v: None if v < 0.5 else 1,
        lambda v: "wide",
    ]
    for f in not_numbers:
        with pytest.raises(TypeError, match="must return numbers"):
            spline.evaluate_profile(f, u, knot, 2)


def test_fast_layer_over_white_matches_km_layer_exactly():
    # solve_absorption's inner loop must give K bit for bit as before
    rng = np.random.default_rng(3)
    for _ in range(200):
        kappa = np.exp(rng.uniform(-14.0, 12.0, sp.N_BANDS))
        kappa[:3] = [0.0, 1e-9, 5e-5]  # the small-B branch
        sigma = float(rng.choice([0.0, 1e-7, rng.uniform(0.0, 15.0)]))
        np.testing.assert_array_equal(
            sp._layer_over_white(kappa, sigma), sp.km_layer(kappa, sigma, 1.0)
        )


# --------------------------------------------------------------------------
# tooling


@pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("git") is None,
    reason="needs make and git",
)
def test_make_spell_changed_sees_committed_and_non_ascii_files(tmp_path):
    makefile = os.path.join(os.path.dirname(__file__), "..", "Makefile")
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    repo.mkdir()
    bin_dir.mkdir()
    # a stand-in for cspell: write down the files it is given
    npx = bin_dir / "npx"
    npx.write_text('#!/bin/sh\ncat > "$CHECKED"\n')
    npx.chmod(0o755)
    (repo / "node_modules" / ".bin").mkdir(parents=True)
    (repo / "node_modules" / ".bin" / "cspell").touch()
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        CHECKED=str(tmp_path / "checked.txt"),
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
        GIT_CONFIG_GLOBAL=os.devnull,
    )

    def run(*args):
        subprocess.run(args, cwd=repo, env=env, check=True, capture_output=True)

    def checked():
        run("make", "-s", "-f", makefile, "spell-changed")
        return sorted((tmp_path / "checked.txt").read_text().splitlines())

    run("git", "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text("node_modules/\n")
    run("git", "add", ".gitignore")
    run("git", "commit", "-q", "-m", "start")
    run("git", "checkout", "-q", "-b", "work")
    (repo / "café.md").write_text("committed\n")
    run("git", "add", "café.md")
    run("git", "commit", "-q", "-m", "work")
    (repo / "staged.md").write_text("staged\n")
    run("git", "add", "staged.md")
    (repo / "ünt.md").write_text("untracked\n")
    assert checked() == ["café.md", "staged.md", "ünt.md"]


# --------------------------------------------------------------------------
# the engine's guards


def _wet_band(surface=None) -> Canvas:
    c = Canvas(200, 120, seed=0, surface=surface)
    c.stroke([(20, 60), (180, 60)], P.ULTRAMARINE_BLUE, amount=30, width=40)
    return c


def _sound(c: Canvas, *brushes_used: Brush, floor: float = -1e-6) -> None:
    """Everything finite, and no wet paint below about zero."""
    for a in (c.top, c.und, c.top_a, c.und_a):
        assert np.isfinite(a).all()
    for b in brushes_used:
        assert all(np.isfinite(layer).all() for layer in b._layers())
    assert min(c.top_v.min(), c.und_v.min()) > floor


def test_a_hard_drag_moves_no_more_paint_than_is_there():
    # drag 0.8 at pressure 0.8 is inside the documented range: the share of
    # each pixel's paint it moved went past 1, and the paint below 0 grew
    # step by step to -1.9e32 (and NaN in the render)
    c = _wet_band()
    brush = c.stroke([(30, 60), (170, 60)], P.CADMIUM_RED, pressure=0.8, drag=0.8)
    _sound(c, brush)
    assert np.isfinite(c.render()).all()


@pytest.mark.parametrize("surface", [surfaces.GLASS, surfaces.COPPER])
@pytest.mark.parametrize("tool", [brushes.STIPPLER, brushes.BRIGHT])
def test_preset_brushes_pressed_hard_on_a_slick_surface_keep_paint_positive(
    surface, tool
):
    c = Canvas(300, 120, seed=0, surface=surface)
    for x in range(60, 260, 40):
        c.stroke(
            [(x, 20), (x, 100)],
            P.ULTRAMARINE_BLUE,
            amount=60,
            brush=brushes.LINER,
            seed=x,
        )
    brush = c.stroke([(30, 60), (270, 60)], brush=tool.sized(40), pressure=1.0, seed=1)
    _sound(c, brush)


def test_swapping_paint_between_layers_exchanges_at_most_all_of_it():
    def layers(rate):
        tip = np.ones((eng.COLUMNS, eng.PACKED))
        belly = np.full((eng.COLUMNS, eng.PACKED), 2.0)
        belly[:, 0] = 0.0
        tip[:, eng.IV] = belly[:, eng.IV] = 1.0
        eng.swap_layers(tip, belly, rate)
        return tip, belly

    full = layers(1.0)
    for rate in (1.6, 40.0):  # a long step of a narrow brush asks for more
        for got, want in zip(layers(rate), full, strict=True):
            np.testing.assert_array_equal(got, want)
    assert min(a.min() for a in full) >= 0


@pytest.mark.skipif(
    bool(os.environ.get("NUMBA_DISABLE_JIT")), reason="nothing is compiled or cached"
)
def test_kernels_cached_for_another_band_count_are_refused(tmp_path):
    # The band count is compiled into the cached kernels, and numba only
    # throws a cached kernel away when its own file changes: changing
    # spectral.WAVELENGTHS alone used to corrupt memory.
    package = tmp_path / "oilpaint"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "..", "oilpaint"),
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    env = dict(
        os.environ,
        NUMBA_CACHE_DIR=str(tmp_path / "cache"),
        PYTHONPATH=str(tmp_path),
        PYTHONDONTWRITEBYTECODE="1",
    )

    def bands():
        return subprocess.run(
            [sys.executable, "-c", "import oilpaint._engine as e; print(e.N)"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

    first = bands()
    assert first.returncode == 0 and first.stdout.split() == ["24"], first.stderr
    spectral = package / "spectral.py"
    text = spectral.read_text()
    assert "np.linspace(400.0, 700.0, 24)" in text
    spectral.write_text(
        text.replace("np.linspace(400.0, 700.0, 24)", "np.linspace(400.0, 700.0, 16)")
    )
    stale = bands()
    assert stale.returncode == 1
    assert "numba cache is stale" in stale.stderr
    shutil.rmtree(tmp_path / "cache")
    fresh = bands()
    assert fresh.returncode == 0 and fresh.stdout.split() == ["16"], fresh.stderr


# --------------------------------------------------------------------------
# footprint tables


def test_footprint_tables_do_not_depend_on_what_was_painted_before():
    # Tables are kept for the aspect rounded to 3 decimals; they used to be
    # built from whichever unrounded aspect asked first.
    tables = []
    for order in ((0.2204, 0.2196), (0.2196, 0.2204)):
        bm._chord_table_at.cache_clear()
        tables.append([bm._chord_table("flat", a, 0.37) for a in order])
    bm._chord_table_at.cache_clear()
    exact = bm._chord_table("flat", 0.22, 0.37)
    for table in (t for pair in tables for t in pair):
        for got, want in zip(table, exact, strict=True):
            np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------
# brush types and brushes

RED, WHITE = P.CADMIUM_RED, P.TITANIUM_WHITE
NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"width": NAN}, "BrushType width must be a finite number"),
        ({"aspect": INF}, "BrushType aspect must be a finite number"),
        ({"stiffness": NAN}, "BrushType stiffness"),
        ({"drag": NAN}, "BrushType drag"),
        ({"width": "wide"}, "BrushType width must be a number"),
        ({"width": 0}, "width and aspect must be positive"),
        ({"clumps": -2}, "clumps .* must be 0"),
        ({"taper": 1.5}, "taper must be from 0 to 1"),
        ({"taper": "a"}, "taper must be a number from 0 to 1 or a .start, end. pair"),
    ],
)
def test_brush_types_check_their_numbers(changes, message):
    with pytest.raises((ValueError, TypeError), match=message):
        BrushType(**changes)


def test_every_preset_is_a_valid_brush_type_and_teeth_are_counted():
    for tool in brushes.ALL.values():
        assert tool.with_() == tool
    with pytest.raises(ValueError, match="teeth of a comb"):
        brushes.knife("straight", 50, teeth=-1)


def test_a_brush_takes_a_brush_type():
    with pytest.raises(TypeError, match="Brush takes a BrushType"):
        Brush("flat")  # pyright: ignore[reportArgumentType]


def _empty(brush: Brush) -> bool:
    return all(not layer.any() for layer in brush._layers())


@pytest.mark.parametrize("amount", [NAN, INF, -INF])
def test_loading_a_non_finite_amount_is_refused_and_loads_nothing(amount):
    brush = Brush(brushes.FLAT_HOG)
    with pytest.raises(ValueError, match="amount must be a finite number"):
        brush.load(RED, amount)
    with pytest.raises(ValueError, match="amount must be a finite number"):
        brush.load_mix([(RED, 1), (WHITE, 1)], amount)
    assert _empty(brush) and brush.width == brushes.FLAT_HOG.width
    brush.load(RED, 6)  # and the brush still works
    assert brush.amount == pytest.approx(6.0)


def test_load_takes_across_third_and_width_by_keyword():
    # the README's load(paint, amount, across, depth)
    left = Brush(brushes.FLAT_HOG).load(RED, 6, "left")
    cols = left.colors()
    assert cols[0] is not None and cols[-1] is None
    sized = Brush(brushes.FLAT_HOG).load(RED, 6, width=10)
    assert sized.width == 10 and sized.amount == pytest.approx(6.0)
    for width in (0, -3, NAN):
        brush = Brush(brushes.FLAT_HOG)
        with pytest.raises(
            ValueError, match=r"width must be (above 0|a finite number)"
        ):
            brush.load(RED, 6, width=width)
        assert _empty(brush) and brush.width == brushes.FLAT_HOG.width
    with pytest.raises(ValueError, match="across"):
        Brush().load(RED, 6, (NAN, 0.5))
    # width used to be the third argument: say so
    brush = Brush(brushes.FLAT_HOG)
    for third in (30, 30.0, np.float32(30)):
        with pytest.raises(TypeError, match=r"load\(paint, amount, width=30"):
            brush.load(RED, 6, third)
    assert _empty(brush) and brush.width == brushes.FLAT_HOG.width


def test_brush_inspection_and_working_check_their_arguments():
    brush = Brush(brushes.FLAT_HOG).load(RED, 6)
    with pytest.raises(ValueError, match="depth must be 'tip', 'belly', 'heel' or"):
        brush.colors("bogus")
    before = [layer.copy() for layer in brush._layers()]
    for kw in ({"passes": NAN}, {"pressure": INF}):
        with pytest.raises(ValueError, match="must be a finite number"):
            brush.work(**kw)
    for got, want in zip(brush._layers(), before, strict=True):
        np.testing.assert_array_equal(got, want)


def test_load_mix_checks_its_parts():
    bad: list[tuple[Any, type]] = [
        (RED, TypeError),
        ([RED, WHITE], TypeError),
        ([(RED, 1, 2)], TypeError),
        ([(RED, NAN)], ValueError),
        ([(RED, 0)], ValueError),
    ]
    for parts, error in bad:
        with pytest.raises(error, match=r"(pairs|proportion|positive)"):
            Brush().load_mix(parts, 6)
    seeds: list[tuple[Any, type]] = [(-1, ValueError), (1.5, TypeError)]
    for seed, error in seeds:
        with pytest.raises(error, match="seed must be None, a whole number >= 0"):
            Brush().load_mix([(RED, 1)], 6, seed=seed)


def test_load_mix_without_a_seed_is_the_same_each_time():
    def painted():
        c = Canvas(120, 60, seed=3)
        b = Brush(brushes.FLAT_HOG).load_mix([(RED, 1), (WHITE, 1)], 8, mixing=0.2)
        c.stroke([(10, 30), (110, 30)], brush=b)
        return c.top

    np.testing.assert_array_equal(painted(), painted())

    def streaks(**kw):
        brush = Brush(brushes.FLAT_HOG)
        brush.load_mix([(RED, 1), (WHITE, 1)], 8, mixing=0.2, **kw)
        return brush.tip + brush.reservoir

    # ... the streaks follow from the load, and from what the brush holds
    assert not np.array_equal(streaks(), streaks(across=(0.0, 0.9)))
    twice = Brush(brushes.FLAT_HOG)
    first = twice.load_mix([(RED, 1), (WHITE, 1)], 8, mixing=0.2).reservoir.copy()
    second = twice.load_mix([(RED, 1), (WHITE, 1)], 8, mixing=0.2).reservoir - first
    assert not np.allclose(first, second)
    # an explicit seed still lays them out by itself
    np.testing.assert_array_equal(streaks(seed=5), streaks(seed=5))
    assert not np.array_equal(streaks(seed=5), streaks(seed=6))


# --------------------------------------------------------------------------
# what a stroke loads


def _loaded(paint, amount=6.0) -> np.ndarray:
    brush = Brush(brushes.FLAT_HOG)
    bm.load_paint(brush, paint, amount)
    return np.stack(brush._layers())


@pytest.mark.parametrize("colour", [[0.8, 0.1, 0.1], [204, 26, 26], (0.8, 0.1, 0.1)])
def test_a_list_of_three_numbers_is_a_colour(colour):
    np.testing.assert_array_equal(_loaded(colour), _loaded(Paint(tuple(colour))))
    c = Canvas(80, 40, seed=1)
    c.stroke([(10, 20), (70, 20)], colour)
    assert c.thickness.max() > 0


def test_loads_may_be_a_list_or_a_tuple_and_keep_their_own_amounts():
    loads = [(RED, 6), (WHITE, 2, "left"), (P.CADMIUM_YELLOW, 0.5, None, "tip")]
    as_list = _loaded(loads)
    np.testing.assert_array_equal(_loaded(tuple(loads)), as_list)
    # (the stroke's amount is not used: each load has its own)
    np.testing.assert_array_equal(_loaded(loads, amount=30), as_list)
    as_dicts = [
        {"paint": RED, "amount": 6},
        {"paint": WHITE, "amount": 2, "across": "left"},
        {"paint": P.CADMIUM_YELLOW, "amount": 0.5, "depth": "tip"},
    ]
    np.testing.assert_array_equal(_loaded(as_dicts), as_list)


def test_a_load_given_as_a_dict_may_say_the_width_it_is_measured_for():
    # (as Brush.load's keywords; it sets the brush's width, as it always did)
    brush = Brush(brushes.FLAT_HOG)
    bm.load_paint(brush, [{"paint": RED, "amount": 6, "width": 40}], 6.0)
    assert brush.width == 40 and brush.amount == pytest.approx(6.0)
    as_kw = Brush(brushes.FLAT_HOG).load(RED, 6, width=40)
    np.testing.assert_array_equal(brush.reservoir, as_kw.reservoir)
    with pytest.raises(ValueError, match="width of load 0 must be above 0"):
        bm.load_paint(brush, [{"paint": RED, "amount": 6, "width": 0}], 6.0)


@pytest.mark.parametrize("colour", [(0.8, 0.1, 0.1, 1.0), [0.8, 0.1], (200,)])
def test_a_colour_has_three_components(colour):
    with pytest.raises(TypeError, match="a colour has three components"):
        bm.Stroke(PATH, colour)


@pytest.mark.parametrize(
    "paint, error, message",
    [
        (["#ff0000"], TypeError, r"load 0 is '#ff0000'; a load is \(paint, amount"),
        ([(RED, 6, None, "tip", 9)], TypeError, "load 0 is .*; a load is"),
        ([(RED,)], TypeError, "load 0 is"),
        ([{"paint": RED, "amount": 6, "size": 9}], ValueError, "has the key 'size'"),
        ([{"paint": RED}], ValueError, "misses paint or amount"),
        ([(RED, 6), (WHITE, NAN)], ValueError, "amount of load 1 must be a finite"),
        ([(RED, 6), (WHITE, 2, "top")], ValueError, "unknown side 'top'"),
        ([(RED, 6), (WHITE, 2, None, "handle")], ValueError, "depth must be"),
        ([(RED, 6), ("not a colour", 2)], ValueError, "cannot parse color"),
    ],
)
def test_bad_loads_are_refused_before_any_paint_goes_on(paint, error, message):
    brush = Brush(brushes.FLAT_HOG)
    with pytest.raises(error, match=message):
        bm.load_paint(brush, paint, 6.0)
    assert _empty(brush) and brush.width == brushes.FLAT_HOG.width
    # ... and a stroke refuses them before it starts
    c = Canvas(120, 60, seed=1)
    before = _state(c, brush)
    with pytest.raises(error, match=message):
        c.stroke([(10, 30), (110, 30)], paint, brush=brush, width=10)
    _unchanged(before, _state(c, brush))
    assert brush.width == brushes.FLAT_HOG.width


# --------------------------------------------------------------------------
# strokes

PATH = [(30, 60), (170, 60)]


def _state(c: Canvas, *tools: Brush) -> list:
    """What painting changes: the canvas's paint and random state, and the
    paint on the brushes."""
    arrays = [c.top, c.und, c.top_a, c.und_a, c.top_id, c.dry_height]
    rng = c._rng.bit_generator.state
    return (
        [a.copy() for a in arrays]
        + [rng, c._stroke_counter]
        + [layer.copy() for b in tools for layer in b._layers()]
    )


def _unchanged(before: list, after: list) -> None:
    for a, b in zip(before, after, strict=True):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"pressure": NAN}, "pressure must be finite, got nan"),
        ({"pressure": [0.2, NAN]}, r"pressure must be finite, got nan \(item 1\)"),
        ({"width": NAN}, "width must be finite and not negative"),
        ({"width": INF}, "width must be finite and not negative"),
        ({"angle": NAN}, "angle must be finite"),
        ({"roll": INF}, "roll must be finite"),
        ({"amount": NAN}, "amount must be a finite number"),
        ({"amount": INF}, "amount must be a finite number"),
        ({"mixing": NAN}, "mixing must be a finite number"),
        ({"drag": INF}, "drag must be a finite number"),
        ({"lift": NAN}, "lift must be a finite number"),
    ],
)
def test_non_finite_stroke_arguments_are_refused_before_anything_changes(kw, message):
    c = _wet_band()
    brush = Brush(brushes.FLAT_HOG).load(WHITE, 4)
    before = _state(c, brush)
    with pytest.raises(ValueError, match=message):
        c.stroke(PATH, RED, brush=brush, **kw)
    _unchanged(before, _state(c, brush))


def _nan_between(u):
    """NaN except at the ends of the stroke."""
    return np.where((u == 0) | (u == 1), 12.0, NAN)


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"pressure": lambda u: u * NAN}, "pressure must be finite along the stroke"),
        ({"width": lambda u: np.full_like(u, INF)}, "width must be finite along"),
        ({"width": lambda u: NAN if u > 0.5 else 3.0}, "width must be finite along"),
        ({"width": _nan_between}, "width must be finite along the stroke"),
        ({"angle": lambda u: np.where(u > 0.5, NAN, 0.0)}, "angle must be finite"),
        ({"roll": lambda u: INF}, "roll must be finite along the stroke"),
    ],
)
def test_a_function_of_u_is_checked_when_painted_before_any_paint(kw, message):
    # (it is only called when the stroke is painted, as often as before: an
    # unseeded stroke has drawn its seed from the canvas by then)
    profile = next(iter(kw.values()))
    stroke = bm.Stroke(PATH, RED, **kw)
    assert getattr(stroke, next(iter(kw))) is profile
    c = _wet_band()
    brush = Brush(brushes.FLAT_HOG).load(WHITE, 4)
    before = _state(c, brush)
    with pytest.raises(ValueError, match=message):
        c.stroke(PATH, RED, brush=brush, **kw)
    after = _state(c, brush)
    del before[6], after[6]  # (the random state)
    _unchanged(before, after)
    assert brush.width == brushes.FLAT_HOG.width


def test_a_function_of_u_is_called_only_to_paint():
    calls = []

    def width(u):
        calls.append(u)
        return 30 if u < 0.5 else 10

    stroke = bm.Stroke(PATH, RED, width=width)
    dataclasses.replace(stroke, pressure=0.4)
    assert calls == []
    Canvas(200, 120, seed=0).paint(stroke)
    assert calls


@pytest.mark.parametrize(
    "kw, error, message",
    [
        ({"brush": "flat"}, TypeError, "brush must be a BrushType .* got 'flat'"),
        ({"brush": brushes}, TypeError, "brush must be a BrushType .* got <module"),
        ({"width": -5}, ValueError, "width must be finite and not negative"),
        ({"width": [30, -1, 10]}, ValueError, r"not negative, got -1.0 \(item 1\)"),
        ({"width": "wide"}, TypeError, "width must be a number, a sequence"),
        ({"pressure": None}, TypeError, "pressure must be a number"),
        ({"pressure": [[0.2, 0.3]]}, ValueError, "pressure must be .* shape"),
        ({"taper": "a"}, TypeError, "taper must be a number from 0 to 1 or a"),
        ({"taper": (0.2, 0.3, 0.4)}, TypeError, "taper must be a number"),
        ({"taper": 1.5}, ValueError, "taper must be from 0 to 1"),
        ({"taper": (0.2, -0.1)}, ValueError, "taper must be from 0 to 1"),
        ({"seed": -1}, ValueError, "seed must be None, a whole number >= 0, a"),
        ({"seed": 1.5}, TypeError, "seed must be None, a whole number >= 0, a"),
        ({"seed": [3, -1]}, ValueError, "seed must be None, a whole number"),
        ({"seed": "s"}, TypeError, "seed must be None, a whole number"),
        ({"mixing": "lots"}, TypeError, "mixing must be a number"),
    ],
)
def test_wrong_stroke_arguments_say_what_is_accepted(kw, error, message):
    c = Canvas(200, 120, seed=0)
    before = _state(c)
    with pytest.raises(error, match=message):
        c.stroke(PATH, RED, **kw)
    _unchanged(before, _state(c))


def test_widths_of_zero_are_drawn_one_pixel_wide():
    # as before: a stroke tapering to nothing is a natural thing to write
    def painted(width):
        c = Canvas(200, 120, seed=0)
        brush = c.stroke(PATH, RED, width=width, seed=3)
        _sound(c, brush)
        return c.top

    np.testing.assert_array_equal(painted(0), painted(1))
    np.testing.assert_array_equal(painted(0.4), painted(1))
    np.testing.assert_array_equal(painted(Fraction(20)), painted(20))
    for tapering in ([20, 10, 0], lambda u: 20 * (1 - u)):
        top = painted(tapering)
        assert top[..., eng.IV][:, 100:].max() > 0


@pytest.mark.parametrize(
    "seed",
    [
        7,
        np.int64(7),
        [7, 11],
        np.array([7, 11], np.uint32),
        np.random.SeedSequence(7),
        np.random.PCG64(7),
    ],
)
def test_seeds_are_anything_default_rng_takes(seed):
    def same(s) -> Any:  # (a fresh copy: a bit generator is used up as it goes)
        return copy.deepcopy(s)

    def generator(s) -> Any:
        return np.random.default_rng(same(s))

    c = Canvas(120, 60, seed=same(seed))
    ref = Canvas(120, 60, seed=generator(seed))
    np.testing.assert_array_equal(c.ground, ref.ground)
    for canvas in (c, ref):
        canvas.stroke(PATH, RED, seed=same(seed))
        canvas.spatter((60, 30), RED, amount=0.2, seed=same(seed))
    _unchanged(_state(c), _state(ref))
    a = Brush().load_mix([(RED, 1), (WHITE, 1)], 6, mixing=0.2, seed=same(seed))
    b = Brush().load_mix([(RED, 1), (WHITE, 1)], 6, mixing=0.2, seed=same(seed))
    np.testing.assert_array_equal(a.reservoir, b.reservoir)
    Canvas(20, 20, seed=generator(seed))  # a Generator too


def test_a_stroke_object_checks_its_arguments_when_made():
    with pytest.raises(ValueError, match="pressure must be finite"):
        bm.Stroke(PATH, RED, pressure=NAN)
    with pytest.raises(TypeError, match="brush must be a BrushType"):
        bm.Stroke(PATH, RED, brush="round")  # pyright: ignore[reportArgumentType]


def test_stroke_physics_overrides_are_clipped_like_a_brush_types():
    tool = brushes.FLAT_HOG
    for name, value, clipped in (
        ("mixing", -1.0, 0.0),
        ("pickup", -1.0, 0.0),
        ("drag", 9.0, 2.0),
        ("streaks", 5.0, 2.0),
        ("blending", 30.0, 1.0),
        ("lift", 3.0, 1.0),
        ("smear", -2.0, 0.0),
    ):
        override: dict[str, Any] = {name: value}
        _, _, eff = bm._resolve(bm.Stroke(PATH, brush=tool, **override))
        assert eff[name] == clipped
        if name not in ("smear", "lift"):
            assert tool.with_(**{name: value}).physics()[name] == clipped
    # in range, and not given: as they were
    _, _, eff = bm._resolve(bm.Stroke(PATH, brush=tool, drag=0.8))
    phys = tool.physics()
    assert eff["drag"] == 0.8 and eff["mixing"] == phys["mixing"]
    for wild in ({"mixing": -1}, {"pickup": -1}, {"blending": 30}, {"drag": 50}):
        c = _wet_band()
        brush = c.stroke([(30, 60), (170, 60)], None, pressure=0.7, **wild)
        _sound(c, brush)


# --------------------------------------------------------------------------
# canvases: making one, stippling, spatter, reading a colour, copying


@pytest.mark.parametrize(
    "args, kw, error, message",
    [
        ((10.7, 5), {}, ValueError, "canvas width must be a whole number of pixels"),
        ((10, 0), {}, ValueError, "canvas height must be a whole number of pixels"),
        ((-4, 5), {}, ValueError, "canvas width must be a whole number"),
        ((NAN, 5), {}, ValueError, "width must be a finite number"),
        (("10", 5), {}, TypeError, "width must be a number"),
        ((20, 20), {"texture": -1}, ValueError, "texture must be 0 .flat. or more"),
        ((20, 20), {"texture": NAN}, ValueError, "texture must be a finite number"),
        ((20, 20), {"weave": 0}, ValueError, "weave must be above 0"),
        ((20, 20), {"seed": -1}, ValueError, "seed must be None, a whole number >= 0"),
        ((20, 20), {"seed": 1.5}, TypeError, "seed must be None, a whole number >= 0"),
    ],
)
def test_canvas_arguments_say_what_is_accepted(args, kw, error, message):
    with pytest.raises(error, match=message):
        Canvas(*args, **kw)


def test_whole_sizes_are_accepted_as_floats():
    size: Any = (12.0, np.int64(8))
    c = Canvas(*size, seed=np.uint8(3))  # pyright: ignore[reportArgumentType]
    assert (c.width, c.height) == (12, 8)


def test_stipple_checks_before_it_loads_the_brush():
    c = Canvas(60, 60, seed=1)
    brush = Brush(brushes.STIPPLER)
    pts = [(10, 10), (20, 20), (30, 30)]
    bad: list[tuple[dict[str, Any], type, str]] = [
        ({"brush": "x"}, TypeError, "brush must be a BrushType"),
        ({"pressure": [0.2, 0.5]}, ValueError, r"one number or one per point \(3\)"),
        ({"pressure": [0.2, NAN, 0.3]}, ValueError, "pressure must be finite"),
        ({"width": 0}, ValueError, "width must be above 0"),
        ({"width": NAN}, ValueError, "width must be a finite number"),
        ({"amount": NAN}, ValueError, "amount must be a finite number"),
        ({"amount": NAN, "width": 9}, ValueError, "amount must be a finite number"),
        ({"width": 9, "paint": [(RED, 2, "top")]}, ValueError, "unknown side"),
    ]
    for kw, error, message in bad:
        kw.setdefault("brush", brush)
        before = _state(c, brush)
        paint = kw.pop("paint", RED)
        with pytest.raises(error, match=message):
            c.stipple(pts, paint, **kw)
        _unchanged(before, _state(c, brush))
        assert brush.width == brushes.STIPPLER.width
    with pytest.raises(ValueError, match="points must be finite"):
        c.stipple([(10, 10), (NAN, 3)], RED)
    with pytest.raises(ValueError, match=r"points must be \(x, y\) points"):
        c.stipple([1, 2, 3], RED)
    assert c.stipple([], RED, brush=brush) is brush  # nothing to tap: fine


def test_spatter_checks_its_arguments():
    c = _wet_band()
    bad: list[tuple[Any, dict[str, Any], type, str]] = [
        ([], {}, ValueError, r"at needs at least one \(x, y\) point"),
        ((NAN, 3), {}, ValueError, "at must be finite"),
        ((10, 10), {"reach": 0}, ValueError, "reach must be above 0"),
        ((10, 10), {"size": -1}, ValueError, "size must be above 0"),
        ((10, 10), {"amount": NAN}, ValueError, "amount must be a finite number"),
        ((10, 10), {"direction": INF}, ValueError, "direction must be a finite"),
        ((10, 10), {"spread": NAN}, ValueError, "spread must be a finite"),
        ((10, 10), {"seed": -2}, ValueError, "seed must be None"),
    ]
    before = _state(c)
    for at, kw, error, message in bad:
        with pytest.raises(error, match=message):
            c.spatter(at, RED, **kw)
    _unchanged(before, _state(c))


def test_spatter_with_no_paint_lays_nothing_and_says_so():
    c = _wet_band()
    before = _state(c)
    assert c.spatter((100, 60), None, seed=1) == 0
    assert c.spatter((100, 60), None) == 0
    _unchanged(before, _state(c))
    assert c.spatter((100, 60), RED, seed=1) > 0


def test_spatter_uses_its_numbers_as_given(monkeypatch):
    # (checked, not turned into Python floats: numpy float32 arguments
    # paint exactly as they always did)
    seen = []
    drop = Canvas._drop

    def recorded(self, x, y, d, stretch, *rest):
        seen.append((type(ux := rest[0]), type(stretch)))
        del ux
        return drop(self, x, y, d, stretch, *rest)

    monkeypatch.setattr(Canvas, "_drop", recorded)
    c = Canvas(120, 60, seed=1)
    f32: Any = np.float32
    c.spatter(
        (60, 30), RED, amount=f32(0.1), direction=f32(30), spread=f32(20),
        reach=f32(40), size=f32(2), seed=4,
    )  # fmt: skip
    assert seen and all(ux is np.float32 for ux, _ in seen)
    assert any(st is np.float32 for _, st in seen)


def test_color_at_takes_pixels_on_the_canvas_only():
    glass = Canvas(30, 20, seed=1, surface=surfaces.GLASS, reverse=True)
    for c in (Canvas(30, 20, seed=1), glass):
        at: Any = (np.int64(29), 19.0)
        assert c.color_at(29, 19) == c.color_at(*at)
        for x, y in ((-1, -1), (30, 0), (0, 20), (0, -1)):
            with pytest.raises(ValueError, match="must be from 0 to"):
                c.color_at(x, y)
        with pytest.raises(TypeError, match="x must be a whole number"):
            c.color_at(1.5, 2)  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="y must be a number"):
            c.color_at(1, "2")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("make_copy", [Canvas.copy, copy.copy, copy.deepcopy])
def test_a_copied_canvas_is_independent_and_paints_on_alike(make_copy):
    c = _wet_band()
    twin = make_copy(c)
    assert isinstance(twin, Canvas) and twin.top is not c.top
    twin.dry()  # (a shallow copy used to dry the original too)
    assert c.wet_thickness.max() > 1 and twin.wet_thickness.max() == 0
    again, other = make_copy(c), make_copy(c)
    for canvas in (again, other):  # unseeded: the copies' own random numbers
        canvas.stroke([(20, 30), (180, 90)], WHITE, brush=brushes.FLAT_HOG)
    _unchanged(_state(again), _state(other))
    assert not np.array_equal(again.top, c.top)
