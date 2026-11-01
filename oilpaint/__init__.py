"""oilpaint: a small oil-painting simulator with physically based color mixing.

Quick start::

    from oilpaint import Canvas, Paint

    canvas = Canvas(800, 600)
    blue = Paint("#1d3f9a")
    canvas.stroke([(100, 300), (300, 220), (500, 320)], paint=blue, amount=8, width=40)
    canvas.stroke([(300, 150), (320, 300), (280, 450)], paint="#f2c80f", amount=6, width=30)
    canvas.save("painting.png")
"""

from . import brushes, palette, spectral, surfaces
from .brush import Brush, BrushType, Stroke
from .canvas import Canvas
from .paint import Paint, mix_colors
from .spline import Bezier, CatmullRom, Polyline
from .surface import Surface

__all__ = [
    "Canvas", "Paint", "Stroke", "Brush", "BrushType", "CatmullRom", "Bezier", "Polyline",
    "mix_colors", "brushes", "palette", "spectral", "Surface", "surfaces",
]
__version__ = "0.1.0"
