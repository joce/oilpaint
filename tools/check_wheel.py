"""Verify the installed wheel: it is imported from the environment (not the
source checkout), it knows its version, and its compiled engine paints.

Run through ``uv run tox -e wheel``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import oilpaint
from oilpaint import Canvas, palette

assert Path(oilpaint.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path.cwd().joinpath("pyproject.toml").exists()
assert oilpaint.__version__ != "0.0.0+unknown", oilpaint.__version__
for name in oilpaint.__all__:
    assert getattr(oilpaint, name) is not None, name

# a red stroke across a small canvas: the numba engine compiles and paints
canvas = Canvas(120, 80, seed=1)
canvas.stroke([(10, 40), (110, 40)], paint=palette.CADMIUM_RED, amount=6, width=20)
image = canvas.render()
assert image.shape == (80, 120, 3), image.shape
r, g, b = image[40, 60]
assert r > g + 0.2 and r > b + 0.2, (r, g, b)
print(f"oilpaint {oilpaint.__version__}: the installed wheel imports and paints.")
