"""A study after Georges Seurat's *Gray Weather, Grande Jatte* (about 1886-88).

Painted from a photograph of the painting (The Metropolitan Museum of Art,
New York; public domain; ``examples/reference``) the way Seurat built his
pointillist pictures: a thin lay-in of the main colours, then, once it is
dry, thousands of small touches of nearly pure colour, each only tinted with
white, set side by side so that they mix in the eye rather than on the
palette. Each dot takes the colour the photograph shows where it falls (his
own dots, at this size), made a little purer, with now and then a dot of its
complement. The dots are tapped on with `canvas.stipple`, bigger ones first
and smaller ones over them. His painted border of dots, darker and bluer than
the picture, is part of it.

    python examples/seurat_grande_jatte.py [output.png]
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
REF = os.path.join(HERE, "reference", "seurat_gray_weather_grande_jatte.jpg")
W = 1000
rng = np.random.default_rng(1886)


class Dots:
    """The paint for each dot: the colour the photograph shows there, a little
    purer (the dots are little more than pure colour tinted with white), and
    now and then its complement. One paint per small step of colour."""

    def __init__(self):
        self.paints = {}

    def dot(self, rgb):
        rgb = np.asarray(rgb, float)
        rgb = np.clip(rgb.mean() + 1.3 * (rgb - rgb.mean()), 0, 1)  # purer
        if rng.random() < 0.04:  # its complement, as a contrast
            rgb = np.clip(rgb.mean() * 2 - rgb, 0, 1) * 0.5 + rgb * 0.5
        key = tuple(np.round(rgb * 18).astype(int))
        if key not in self.paints:
            self.paints[key] = Paint(np.asarray(key) / 18, body=0.8)
        return self.paints[key]


def lay_in(c, ref):
    """A thin lay-in of the main colours, brushed loosely."""
    step = 16
    brush = brushes.FLAT_HOG.sized(30)
    for y in np.arange(step / 2, c.height, step):
        for x in np.arange(step / 2, c.width, step):
            jx, jy = x + rng.uniform(-5, 5), y + rng.uniform(-5, 5)
            angle, _ = ref.direction(jx, jy, scale=12)
            d = 14
            pts = [(jx - d * np.cos(angle), jy - d * np.sin(angle)),
                   (jx + d * np.cos(angle), jy + d * np.sin(angle))]  # fmt: skip
            c.stroke(
                pts, ref.color(jx, jy, scale=10), amount=3, brush=brush, pressure=0.4
            )


def dots(c, ref, div, size, spacing, scale):
    """One layer of dots ``size`` px across, about ``spacing`` apart."""
    ys, xs = np.mgrid[0 : c.height : spacing, 0 : c.width : spacing]
    xs = xs.ravel() + rng.uniform(-0.45, 0.45, xs.size) * spacing
    ys = ys.ravel() + rng.uniform(-0.45, 0.45, ys.size) * spacing
    # sort the dots by their paint, then tap each paint's dots in small batches
    # (a brushful at a time), the batches in random order
    by_paint = defaultdict(list)
    names = {}
    for x, y in zip(xs, ys, strict=True):
        paint = div.dot(ref.color(x, y, scale=scale))
        by_paint[id(paint)].append((x, y))
        names[id(paint)] = paint
    batches = []
    for key, pts in by_paint.items():
        order = rng.permutation(len(pts))
        batches.extend(
            (names[key], [pts[i] for i in order[k : k + 8]])
            for k in range(0, len(pts), 8)
        )
    brush = brushes.ROUND.sized(size)
    for k in rng.permutation(len(batches)):
        paint, pts = batches[k]
        c.stipple(pts, paint, amount=1.6 * len(pts) + 2, pressure=0.6, brush=brush)
    return len(xs)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    c = Canvas(W, h, surface=surfaces.CANVAS.with_(ground="#ece6d8"), seed=1886)
    div = Dots()
    print(f"[{time.strftime('%H:%M:%S')}] lay-in", flush=True)
    lay_in(c, ref)
    c.dry()
    n = 0
    for size, spacing, scale in ((7, 6.5, 3), (5, 4.6, 1.5), (4, 5.5, 0.8)):
        print(f"[{time.strftime('%H:%M:%S')}] dots {size}px ({n} so far)", flush=True)
        n += dots(c, ref, div, size, spacing, scale)
        c.dry()
    c.save(path)
    print(f"wrote {path}: {n} dots in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "seurat_grande_jatte.png")
    )
