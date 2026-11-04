"""A study after J. M. W. Turner's *Approach to Venice* (1844).

Painted from a photograph of the painting (National Gallery of Art,
Washington; public domain; ``examples/reference``) the way the late Turner
worked: thin colour laid in loosely on a white ground, the haze swept across
it with the side of the hand (the `PALM`), the colour pushed on and rubbed
along the forms with the fingers (the `FINGER`, loaded with paint) until the
sky and the water dissolve in light; then the city, the boats and their
reflections touched in with a fingertip, a few of the touches pressed so it
leaves its print, and a small brush kept for the masts. Each layer looks at
the canvas so far and paints only where it still differs from the photograph
(see ``_studies``).

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
    rows = np.arange(y0, y1, 28)
    for y in rows:
        there = [
            (rng.uniform(-40, 40), y),
            (c.width * 0.5, y + rng.uniform(-6, 6)),
            (c.width + 40, y + 4),
        ]
        back = [(c.width + 40, y + 12), (c.width * 0.4, y + 10 + rng.uniform(-6, 6))]
        c.stroke(there, brush=brushes.PALM, pressure=0.4)
        c.stroke(back, brush=brushes.PALM, pressure=0.35)
    return 2 * len(rows)


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
    brush = study.layer(64, brushes.FLAT_SABLE, straight=True, load=0.5, pressure=0.4)
    log("the palm across the sky")
    sweeps = palm(study, 0, 0.5 * h)
    log("colour worked in with the fingers")
    finger = study.layer(30, brushes.FINGER, threshold=5.0, load=0.5, pressure=0.4)
    log("rubbed along the forms")
    rubs = fingers(study, soft)
    c.dry()
    log("the city, the boats, the reflections, with a fingertip")
    finger += study.layer(14, brushes.FINGER, threshold=7.0, load=0.6, pressure=0.4,
                          where=~soft)  # fmt: skip
    # a brush only for what a finger cannot draw: the masts, the finest touches
    brush += study.layer(6, brushes.ROUND, threshold=9.0, length=3, load=0.7,
                         pressure=0.45, where=~soft)  # fmt: skip
    log("fingertips")
    prints = fingertips(study, ~soft)
    c.save(path, relief=0.4, gloss=0.5)
    print(
        f"wrote {path}: {finger} finger strokes, {rubs} finger rubs, {sweeps} palm "
        f"sweeps, {prints} fingertip prints and {brush} brush strokes, "
        f"in {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "turner_venice.png")
    )
