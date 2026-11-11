"""The public API as tools see it: type hints that resolve, and the names of
oilpaint.brush, the stroke's included; what the presets list; the tables
shared between modules; and what a stroke uses of a canvas."""

import importlib
import inspect
import typing

import numpy as np
import pytest

import oilpaint
from oilpaint import Paint, Surface, brushes, palette, surfaces
from oilpaint import brush as bm
from oilpaint import stroke as sm
from oilpaint.brush import BrushType

PUBLIC = [
    oilpaint.Stroke,
    oilpaint.Brush,
    oilpaint.BrushType,
    oilpaint.Canvas,
    oilpaint.Paint,
    oilpaint.Surface,
    oilpaint.Reference,
]


def _public_functions():
    for cls in PUBLIC:
        yield f"{cls.__name__}", cls
        for name, member in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") or name == "__init__":
                yield f"{cls.__name__}.{name}", member
    yield "apply_stroke", sm.apply_stroke
    yield "brush.apply_stroke", bm.apply_stroke


@pytest.mark.parametrize(("name", "target"), list(_public_functions()))
def test_type_hints_of_the_public_api_resolve(name, target):
    typing.get_type_hints(target)


def _forward_refs(hint):
    if isinstance(hint, (str, typing.ForwardRef)):
        yield hint
    for arg in typing.get_args(hint):
        yield from _forward_refs(arg)


@pytest.mark.parametrize("name", ["ColorLike", "StrokePaint"])
def test_the_public_type_aliases_hold_no_names_to_look_up(name):
    # A name in an alias is looked up in whichever module's hints use it, and
    # fails there unless that module imports it too; before Python 3.14 it
    # even passed or failed with the order hints were first resolved in (the
    # first lookup was kept).
    assert not list(_forward_refs(getattr(bm, name)))


def test_the_stroke_is_found_in_oilpaint_brush():
    for name in ("Stroke", "apply_stroke"):
        bm.__dict__.pop(name, None)  # (as if not looked up yet)
        assert name in dir(bm)
        assert getattr(bm, name) is getattr(sm, name)
        assert bm.__dict__[name] is getattr(sm, name)  # (kept, once looked up)
    bm.__dict__.pop("Stroke")
    assert dict(inspect.getmembers(bm))["Stroke"] is sm.Stroke
    namespace: dict = {}
    bm.__dict__.pop("apply_stroke")
    exec("from oilpaint.brush import *", namespace)
    assert namespace["Stroke"] is sm.Stroke
    assert namespace["apply_stroke"] is sm.apply_stroke
    for name in ("Brush", "BrushType", "StrokePaint", "load_paint", "np"):
        assert namespace[name] is getattr(bm, name)
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        bm.nothing  # ruff: ignore[useless-expression]  # pyright: ignore[reportAttributeAccessIssue]


def test_import_star_from_oilpaint_brush_gives_what_all_lists():
    namespace: dict = {}
    exec("from oilpaint.brush import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(bm.__all__)
    assert not [name for name in bm.__all__ if name.startswith("_")]


@pytest.mark.parametrize(
    ("module", "kind"),
    [(brushes, BrushType), (palette, Paint), (surfaces, Surface)],
)
def test_all_lists_every_preset_of_its_module(module, kind):
    # (ALL is written out: a preset added to the module must be added to it)
    presets = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, kind) and not name.startswith("_")
    }
    listed = module.ALL
    if isinstance(listed, dict):
        assert listed == presets
    else:
        assert [id(p) for p in listed] == [id(p) for p in presets.values()]


@pytest.mark.parametrize(
    "table",
    [
        "spectral.WAVELENGTHS",
        "spectral._CIE_WL",
        "spectral._CIE_1931",
        "spectral._D65_SPD",
        "spectral._XYZ_TO_LINEAR_SRGB",
        "spectral.SPECTRUM_TO_RGB",
        "spectral.LUMINANCE",
        "spectral._RGB_TO_XYZ",
        "spectral._D65_WHITE_XYZ",
        "spectral._BASIS",
        "_render.TAU_TABLE",
        "palette._STEPS",
        "brush._COL_X",
        "brush._RING_A",
        "brush._RING_X",
        "_geometry._NO_TABLES",
        "_hair._NO_TEX",
        "_hair._NO_TEX2",
    ],
)
def test_shared_tables_are_read_only(table):
    module, name = table.split(".")
    value = getattr(importlib.import_module("oilpaint." + module), name)
    for a in value if isinstance(value, tuple) else (value,):
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 0.5


def test_the_paint_a_knife_holds_back_stays_writable():
    # (a knife holds no paint back and nothing writes to what stands for it,
    # but the kernels that take it write to what a brush holds back, and
    # numba does not compile a write to a read-only array)
    assert all(a.flags.writeable for a in sm._NO_PENDING)
    assert not sm._NO_PENDING[4].any()  # (none held: never "on")


def test_a_canvas_is_what_a_stroke_paints_on():
    c = oilpaint.Canvas(30, 20, seed=1)
    for name in (*sm._Canvas.__annotations__, "top_v", "und_v"):
        assert hasattr(c, name), name
    for name in ("_stroke_rng", "_new_stroke_id", "_relief"):
        assert callable(getattr(c, name)), name
    relief = c._relief()
    assert isinstance(relief, sm._Relief) and c._relief() is relief
    assert all(m.shape == (20, 30) for m in relief)
    # the parts of the wet paint are views of it, and only that
    c.top_v[3, 4] = 2.0
    assert c.top[3, 4, bm.IV] == 2.0 and np.shares_memory(c.und_k, c.und)
    with pytest.raises(AttributeError):
        c.top_v = np.zeros((20, 30))  # pyright: ignore[reportAttributeAccessIssue]
    # drying makes the maps again, from the dried paint
    c.dry()
    assert c._relief() is not relief
    assert c._relief().floor[3, 4] > relief.floor[3, 4]
