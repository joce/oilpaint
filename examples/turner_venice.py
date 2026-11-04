"""A study after J. M. W. Turner's *Approach to Venice* (1844).

Painted from a photograph of the painting (National Gallery of Art,
Washington; public domain; ``examples/reference``) the way the late Turner
worked: thin colour laid in loosely on a white ground, then smudged and
dissolved while wet, as much with his fingers and the side of his hand as
with a brush, until the city floats in light; a few touches of strong colour
on top, softened again with a fingertip. Every mark is a `canvas.stroke`, and
the blending is done with the `PALM` and the `FINGER`.

    python examples/turner_venice.py [output.png]
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "turner_approach_to_venice.jpg")
W = 1000
rng = np.random.default_rng(1844)


_PAINTS: dict = {}


def paint_of(rgb, levels=24):
    """A paint for this colour, one per small step of colour (cached)."""
    key = tuple(np.round(np.clip(rgb, 0, 1) * levels).astype(int))
    if key not in _PAINTS:
        _PAINTS[key] = Paint(np.asarray(key) / levels, body=0.35)
    return _PAINTS[key]


def along(ref, x, y, length, scale):
    """A short straight path through (x, y) following the forms."""
    angle, _ = ref.direction(x, y, scale=scale)
    d = 0.5 * length
    return [(x - d * np.cos(angle), y - d * np.sin(angle)),
            (x + d * np.cos(angle), y + d * np.sin(angle))]  # fmt: skip


def grid(c, step, jitter=0.5):
    ys, xs = np.mgrid[step / 2 : c.height : step, step / 2 : c.width : step]
    xs = xs.ravel() + rng.uniform(-jitter, jitter, xs.size) * step
    ys = ys.ravel() + rng.uniform(-jitter, jitter, ys.size) * step
    order = rng.permutation(xs.size)
    return xs[order], ys[order]


def lay_in(c, ref):
    """Thin colour, loosely, with a big soft flat."""
    brush = brushes.FLAT_SABLE.sized(44)
    for x, y in zip(*grid(c, 22), strict=True):
        paint = paint_of(ref.color(x, y, scale=16))
        c.stroke(along(ref, x, y, 50, 16), paint, amount=2.2, brush=brush, pressure=0.4)


def palm(c, ref, y0, y1):
    """The side of the hand swept across, back and forth: haze."""
    for y in np.arange(y0, y1, 26):
        x0 = rng.uniform(-40, 40)
        there = [
            (x0, y),
            (c.width * 0.5, y + rng.uniform(-6, 6)),
            (c.width + 40, y + 4),
        ]
        back = [(c.width + 40, y + 12), (c.width * 0.4, y + 10 + rng.uniform(-6, 6))]
        c.stroke(there, brush=brushes.PALM, pressure=0.45)
        c.stroke(back, brush=brushes.PALM, pressure=0.4)


def second_layer(c, ref):
    """Smaller touches where the picture has something to say: the city,
    the boats, their reflections."""
    brush = brushes.FILBERT_SABLE.sized(14)
    n = 0
    for x, y in zip(*grid(c, 9), strict=True):
        if ref.detail(x, y, scale=6) < 0.045:
            continue
        paint = paint_of(ref.color(x, y, scale=4))
        c.stroke(along(ref, x, y, 18, 5), paint, amount=4, brush=brush, pressure=0.45)
        n += 1
    return n


def fingers(c, ref):
    """A fingertip rubbed along the forms wherever the picture is soft: the
    edges dissolve."""
    n = 0
    for x, y in zip(*grid(c, 30), strict=True):
        if ref.detail(x, y, scale=8) > 0.05:
            continue
        path = along(ref, x, y, rng.uniform(30, 60), 8)
        finger = brushes.FINGER.sized(rng.uniform(22, 34))
        c.stroke(path, brush=finger, pressure=0.45)  # rubbed one way...
        c.stroke(path[::-1], brush=finger, pressure=0.4)  # ...and back
        n += 1
    return n


def accents(c, ref):
    """The last strong touches, some softened again with a fingertip."""
    n = 0
    for x, y in zip(*grid(c, 7), strict=True):
        if ref.detail(x, y, scale=2.5) < 0.07:
            continue
        rgb = ref.color(x, y, scale=1.2)
        c.stroke(along(ref, x, y, 10, 2.5), paint_of(rgb), amount=5,
                 brush=brushes.ROUND.sized(5), pressure=0.5)  # fmt: skip
        if rng.random() < 0.25:  # pressed with a fingertip: it leaves its print
            c.stroke([(x, y)], brush=brushes.FINGER.sized(18), pressure=0.2)
        n += 1
    return n


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    c = Canvas(W, h, surface=surfaces.CANVAS.with_(ground="#f4f0e4"), seed=1844)
    log("lay-in")
    lay_in(c, ref)
    log("palm")
    palm(c, ref, 0, 0.55 * h)  # the sky
    log("fingers")
    f = fingers(c, ref)
    log("second layer")
    n = second_layer(c, ref)
    log("palm again, lightly, over the sky")
    palm(c, ref, 0, 0.4 * h)
    log("accents")
    a = accents(c, ref)
    c.save(path, relief=0.6)
    print(f"wrote {path}: {n} touches, {f} finger rubs, {a} accents, "
          f"in {time.time() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "turner_venice.png")
    )
