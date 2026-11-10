"""oilpaint: a small oil-painting simulator with physically based color mixing.

Quick start::

    from oilpaint import Canvas, Paint

    canvas = Canvas(800, 600)
    blue = Paint("#1d3f9a")
    canvas.stroke([(100, 300), (300, 220), (500, 320)], paint=blue, amount=8, width=40)
    yellow = "#f2c80f"
    canvas.stroke([(300, 150), (320, 300), (280, 450)], yellow, amount=6, width=30)
    canvas.save("painting.png")
"""

from . import brushes, palette, reference, spectral, surfaces
from ._about import __version__
from .brush import Brush, BrushType
from .canvas import Canvas
from .paint import Paint, mix_colors
from .reference import Reference
from .spline import Bezier, CatmullRom, Polyline
from .stroke import Stroke
from .surface import Surface

__all__ = [
    "Bezier",
    "Brush",
    "BrushType",
    "Canvas",
    "CatmullRom",
    "Paint",
    "Polyline",
    "Reference",
    "Stroke",
    "Surface",
    "__version__",
    "brushes",
    "mix_colors",
    "palette",
    "reference",
    "spectral",
    "surfaces",
]
