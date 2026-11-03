"""The smallest useful example: two crossing strokes and a blend."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint

canvas = Canvas(600, 400, seed=1)

blue = Paint("#23399b", opacity=0.5)  # a semi-transparent ultramarine
yellow = Paint("#f7c600")  # an opaque cadmium yellow

# A stroke is a spline through your points, a paint and an amount of paint.
canvas.stroke([(60, 140), (250, 100), (540, 170)], paint=blue, amount=12, width=50)

# Crossing it while it is still wet: pressure decides how much they churn together.
canvas.stroke(
    [(200, 30), (230, 200), (190, 370)], paint=yellow, amount=10, width=40, pressure=0.2
)
canvas.stroke(
    [(400, 30), (430, 200), (390, 370)], paint=yellow, amount=10, width=40, pressure=0.9
)

# A stroke with no paint is a clean brush: it drags and blends the wet paint.
canvas.stroke([(80, 300), (560, 300)], width=40, pressure=0.6)

out = os.path.join(os.path.dirname(__file__), "output", "quickstart.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
canvas.save(out)
print("wrote", out)
