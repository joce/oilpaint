"""The public API as tools see it: type hints that resolve, and the names of
oilpaint.brush, the stroke's included."""

import inspect
import typing

import pytest

import oilpaint
from oilpaint import brush as bm
from oilpaint import stroke as sm

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
