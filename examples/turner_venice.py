"""A study after J. M. W. Turner's *Approach to Venice* (1844).

Painted from a photograph of the painting (National Gallery of Art,
Washington; public domain; ``examples/reference``) the way the late Turner
worked: thin colour laid in loosely on a white ground and smudged while wet,
as much with his fingers and the side of his hand as with a brush, until the
sky and the water dissolve in light; then the city, the boats and their
reflections touched in over it, a few of the touches softened again with a
fingertip. Each layer looks at the canvas so far and paints only where it
still differs from the photograph (see ``_studies``); the smudging is done
with the `PALM` and the `FINGER`.

    python examples/turner_venice.py [output.png]
"""

import os
import sys
import time

import numpy as np
from _studies import Study
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "turner_approach_to_venice.jpg")
W = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def palm(study, y0, y1):
    """The side of the hand swept across the sky, there and back: haze."""
    c, rng = study.c, study.rng
    for y in np.arange(y0, y1, 28):
        there = [
            (rng.uniform(-40, 40), y),
            (c.width * 0.5, y + rng.uniform(-6, 6)),
            (c.width + 40, y + 4),
        ]
        back = [(c.width + 40, y + 12), (c.width * 0.4, y + 10 + rng.uniform(-6, 6))]
        c.stroke(there, brush=brushes.PALM, pressure=0.4)
        c.stroke(back, brush=brushes.PALM, pressure=0.35)


def fingers(study, soft):
    """A fingertip rubbed along the forms, there and back, wherever the
    picture is soft."""
    n = 0
    for x, y in zip(*study.points(34, soft), strict=True):
        a, _ = study.ref.direction(x, y, scale=8)
        d = 0.5 * study.rng.uniform(30, 55)
        path = [
            (x - d * np.cos(a), y - d * np.sin(a)),
            (x + d * np.cos(a), y + d * np.sin(a)),
        ]
        finger = brushes.FINGER.sized(study.rng.uniform(22, 32))
        study.c.stroke(path, brush=finger, pressure=0.4)
        study.c.stroke(path[::-1], brush=finger, pressure=0.35)
        n += 1
    return n


def fingertips(study, busy):
    """A few of the last touches pressed with a fingertip, which leaves its print."""
    n = 0
    for x, y in zip(*study.points(26, busy), strict=True):
        if study.rng.random() < 0.5:
            study.c.stroke([(x, y)], brush=brushes.FINGER.sized(16), pressure=0.2)
            n += 1
    return n


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    c = Canvas(W, h, surface=surfaces.CANVAS.with_(ground="#f4f0e4"), seed=1844)
    study = Study(c, ref, seed=1844, body=0.35)
    ys, xs = np.mgrid[0:h, 0:W]
    detail = np.asarray(ref.detail(xs, ys, scale=8))
    soft = detail < 0.05
    log("thin colour, laid in loosely")
    study.layer(64, brushes.FLAT_SABLE, straight=True, load=0.5, pressure=0.4)
    log("the palm across the sky")
    palm(study, 0, 0.5 * h)
    log("a second layer")
    study.layer(30, brushes.FILBERT_SABLE, threshold=5.0, load=0.5, pressure=0.4)
    log("fingers")
    f = fingers(study, soft)
    c.dry()
    log("the city, the boats, the reflections")
    study.layer(14, brushes.FILBERT_SABLE, threshold=7.0, load=0.5, pressure=0.45)
    # the small touches only where there is something to draw (not in the haze)
    study.layer(
        7, brushes.ROUND, threshold=7.0, length=3, load=0.7, pressure=0.45, where=~soft
    )
    log("fingertips")
    t = fingertips(study, ~soft)
    study.layer(4, brushes.ROUND, threshold=10.0, length=3, scale=0.8, load=0.9,
                pressure=0.5, where=~soft)  # fmt: skip
    c.save(path, relief=0.4, gloss=0.5)
    print(
        f"wrote {path}: {study.strokes} brush strokes, {f} finger rubs, "
        f"{t} fingertip prints, in {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "turner_venice.png")
    )
