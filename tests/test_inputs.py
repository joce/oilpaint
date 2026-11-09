"""Input handling of the colour, paint, path, palette, reference and surface
helpers: what is accepted, what is refused, and with what message. None of
these need the compiled stroke engine."""

import math
import os
import pickle
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, spline
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
