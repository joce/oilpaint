"""A study after Georges Seurat's *Gray Weather, Grande Jatte* (about 1886-88).

Painted from a photograph of the painting (The Metropolitan Museum of Art,
New York; public domain; ``examples/reference``, without his painted border)
the way Seurat built his pointillist pictures: a thin lay-in of the main
colours, then, once it is dry, thousands of small touches of nearly pure
colour set side by side, so that they mix in the eye rather than on the
palette. Each dot takes the colour the photograph shows where it falls (his
own dots, at this size), made a little purer and nudged towards whatever the
canvas still lacks there; now and then one is its complement. The dots are
tapped on with `canvas.stipple`, bigger ones first, smaller ones over them.

    python examples/seurat_grande_jatte.py [output.png]
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
from _studies import Study
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "seurat_gray_weather_grande_jatte.jpg")
W = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Dots:
    """The paint for each dot: nearly pure colour (one paint per small step
    of colour), now and then its complement."""

    def __init__(self, rng):
        self.rng = rng
        self.paints = {}

    def dot(self, rgb):
        rgb = np.asarray(rgb, float)
        rgb = np.clip(rgb.mean() + 1.3 * (rgb - rgb.mean()), 0, 1)  # purer
        if self.rng.random() < 0.04:  # its complement, as a contrast
            rgb = np.clip(rgb.mean() * 2 - rgb, 0, 1) * 0.5 + rgb * 0.5
        key = tuple(np.round(rgb * 20).astype(int))
        if key not in self.paints:
            self.paints[key] = Paint(np.asarray(key) / 20, body=0.8)
        return self.paints[key]


def dots(study, palette, size, spacing, scale, correct):
    """One layer of dots ``size`` px across, about ``spacing`` apart."""
    target, lacking, _ = study.gap(size, scale)
    xs, ys = study.points(spacing)
    # sort the dots by their paint, then tap each paint's dots in small batches
    # (a brushful at a time), the batches in random order
    by_paint = defaultdict(list)
    paints = {}
    for x, y in zip(xs, ys, strict=True):
        yi, xi = int(y), int(x)
        paint = palette.dot(target[yi, xi] + correct * lacking[yi, xi])
        by_paint[id(paint)].append((x, y))
        paints[id(paint)] = paint
    batches = []
    for key, pts in by_paint.items():
        batches.extend((paints[key], pts[k : k + 8]) for k in range(0, len(pts), 8))
    brush = brushes.ROUND.sized(size)
    for k in study.rng.permutation(len(batches)):
        paint, pts = batches[k]
        study.c.stipple(
            pts, paint, amount=1.6 * len(pts) + 2, pressure=0.5, brush=brush
        )
    return len(xs)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    c = Canvas(W, h, surface=surfaces.CANVAS.with_(ground="#ece6d8"), seed=1886)
    study = Study(c, ref, seed=1886, body=0.5)
    log("lay-in")
    study.layer(44, brushes.FLAT_HOG, load=0.5, pressure=0.4, straight=True)
    study.layer(20, brushes.FILBERT_HOG, threshold=5.0, load=0.5, pressure=0.4)
    c.dry()
    palette = Dots(study.rng)
    n = 0
    for size, spacing, scale, correct in (
        (7, 5.5, 3, 0.4),
        (5, 4.2, 1.5, 0.6),
        (3.5, 3.6, 0.8, 0.7),
    ):
        log(f"dots {size}px ({n} so far)")
        n += dots(study, palette, size, spacing, scale, correct)
        c.dry()
    c.save(path)
    print(
        f"wrote {path}: {study.strokes} lay-in strokes and {n} dots "
        f"in {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "seurat_grande_jatte.png")
    )
