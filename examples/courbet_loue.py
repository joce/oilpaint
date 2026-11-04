"""A study after Gustave Courbet's *The Source of the Loue* (1864).

Painted from a photograph of the painting (The Metropolitan Museum of Art,
New York; public domain; ``examples/reference``) the way Courbet painted his
rocks: on a dark ground, the darks laid in first, then the stone built up
light over dark, not brushed but pressed on with sponges (and rags), whose
uneven prints give the rock its grain, and the falling water put on with the
palette knife. Every mark is a `canvas.stroke`, `canvas.stipple` or knife
stroke; numpy only works out where they go.

    python examples/courbet_loue.py [output.png]
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "courbet_source_of_the_loue.jpg")
W = 1000
rng = np.random.default_rng(1864)


_PAINTS: dict = {}


def quantized(rgb, levels=24):
    """A paint for this colour, one per small step of colour (cached)."""
    key = tuple(np.round(np.clip(rgb, 0, 1) * levels).astype(int))
    if key not in _PAINTS:
        _PAINTS[key] = Paint(np.asarray(key) / levels, body=0.75)
    return _PAINTS[key]


def darker(rgb, k):
    return np.clip(np.asarray(rgb) * k, 0, 1)


def lay_in(c, ref):
    """The whole picture laid in broadly, the darks a shade deeper."""
    step = 14
    brush = brushes.FLAT_HOG.sized(28)
    for y in np.arange(step / 2, c.height, step):
        for x in np.arange(step / 2, c.width, step):
            jx, jy = x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)
            angle, _ = ref.direction(jx, jy, scale=10)
            d = 14
            pts = [(jx - d * np.cos(angle), jy - d * np.sin(angle)),
                   (jx + d * np.cos(angle), jy + d * np.sin(angle))]  # fmt: skip
            paint = quantized(darker(ref.color(jx, jy, scale=10), 0.72))
            c.stroke(pts, paint, amount=4, brush=brush, pressure=0.45)


def sponged(c, ref, size, spacing, scale, lightest, pressure, where=None, lift=1.0):
    """Sponge prints of the picture's colour, about ``spacing`` apart, where it
    is at least ``lightest`` light: the stone built up light over dark."""
    ys, xs = np.mgrid[0 : c.height : spacing, 0 : c.width : spacing]
    xs = xs.ravel() + rng.uniform(-0.5, 0.5, xs.size) * spacing
    ys = ys.ravel() + rng.uniform(-0.5, 0.5, ys.size) * spacing
    groups = defaultdict(list)
    paints = {}
    for x, y in zip(xs, ys, strict=True):
        if ref.value(x, y, scale=scale) < lightest:
            continue
        rgb = ref.color(x, y, scale=scale)
        if where is not None and not where(rgb):
            continue
        # a touch lighter (the dark ground shows through the sponge's holes),
        # and never quite the same twice
        paint = quantized(darker(rgb, lift * rng.uniform(0.9, 1.1)))
        groups[id(paint)].append((x, y))
        paints[id(paint)] = paint
    batches = []
    for key, pts in groups.items():
        batches.extend((paints[key], pts[k : k + 5]) for k in range(0, len(pts), 5))
    sponge = brushes.SPONGE.sized(size)
    for k in rng.permutation(len(batches)):
        paint, pts = batches[k]
        c.stipple(
            pts, paint, amount=3.0 * len(pts) + 4, pressure=pressure, brush=sponge
        )
    return sum(len(p) for _, p in batches)


def greenish(rgb):
    r, g, b = rgb
    return g > r * 0.98 and g > b * 1.05


def water(c, ref):
    """The falling water and the foam: the palette knife, following the flow."""
    knife = brushes.KNIFE_SMALL
    n = 0
    for y in np.arange(c.height * 0.45, c.height, 7):
        for x in np.arange(0, c.width, 9):
            jx, jy = x + rng.uniform(-4, 4), y + rng.uniform(-3, 3)
            if ref.value(jx, jy, scale=2) < 0.62:
                continue
            angle, _ = ref.direction(jx, jy, scale=5)
            d = rng.uniform(6, 14)
            pts = [(jx - d * np.cos(angle), jy - d * np.sin(angle)),
                   (jx + d * np.cos(angle), jy + d * np.sin(angle))]  # fmt: skip
            paint = quantized(ref.color(jx, jy, scale=1.5))
            c.stroke(pts, paint, amount=5, brush=knife, pressure=0.5, angle=90)
            n += 1
    return n


def details(c, ref):
    """The mill's windows and timbers, the cracks in the stone: short dark
    accents, only where the picture is both dark and sharp."""
    n = 0
    for y in np.arange(3, c.height, 6):
        for x in np.arange(3, c.width, 6):
            jx, jy = x + rng.uniform(-2, 2), y + rng.uniform(-2, 2)
            if ref.value(jx, jy, scale=1) > 0.22 or ref.detail(jx, jy, scale=2) < 0.12:
                continue
            angle, _ = ref.direction(jx, jy, scale=2)
            d = 4
            pts = [(jx - d * np.cos(angle), jy - d * np.sin(angle)),
                   (jx + d * np.cos(angle), jy + d * np.sin(angle))]  # fmt: skip
            paint = quantized(ref.color(jx, jy, scale=0.8))
            c.stroke(pts, paint, amount=6, brush=brushes.ROUND.sized(3), pressure=0.5)
            n += 1
    return n


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    # Courbet worked up from a dark brown ground
    c = Canvas(W, h, surface=surfaces.LINEN_FINE.with_(ground="#2e2620"), seed=1864)
    log("lay-in")
    lay_in(c, ref)
    c.dry()
    n = 0
    for size, spacing, scale, lightest, pressure, lift in (
        (32, 17, 3, 0.08, 0.24, 1.12),  # the stone, lighter over the dark lay-in
        (24, 14, 2, 0.3, 0.22, 1.18),  # its lights
        (16, 11, 1.2, 0.5, 0.2, 1.22),  # the brightest faces, barely touched
    ):
        log(f"sponge {size}px ({n} prints so far)")
        n += sponged(c, ref, size, spacing, scale, lightest, pressure, lift=lift)
        c.dry()
    log("foliage")
    n += sponged(c, ref, 18, 10, 2.5, 0.08, 0.35, where=greenish)
    c.dry()
    log("water")
    knife = water(c, ref)
    c.dry()
    log("details")
    marks = details(c, ref)
    c.save(path)
    print(f"wrote {path}: {n} sponge prints, {knife} knife strokes, {marks} details, "
          f"in {time.time() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "courbet_loue.png")
    )
