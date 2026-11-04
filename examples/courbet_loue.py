"""A study after Gustave Courbet's *The Source of the Loue* (1864).

Painted from a photograph of the painting (The Metropolitan Museum of Art,
New York; public domain; ``examples/reference``) the way Courbet painted his
rocks: on a dark ground, the masses laid in thinly with a broad brush, then
the stone built up light over dark almost entirely with sponges, big ones
first and smaller ones over them, their uneven prints, with the dark showing
through their holes, giving it its grain; the falling water laid on with the
palette knife; and only the darkest lines, the cracks, the timbers of the
mill and its windows, drawn with a small brush. Each pass looks at the canvas
so far and prints or paints only where it still differs from the photograph
(see ``_studies``).

    python examples/courbet_loue.py [output.png]
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
from _studies import Study
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "courbet_source_of_the_loue.jpg")
W = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sponged(study, size, spacing, scale, where, pressure, lift=1.05, threshold=3.0):
    """Sponge prints where the canvas still lacks the stone's colour: the
    photograph's colour there, a touch lighter (the darker paint beneath shows
    through the sponge's holes)."""
    target, lacking, err = study.gap(size, scale)
    xs, ys = study.points(spacing, where & (err > threshold))
    groups = defaultdict(list)
    paints = {}
    for x, y in zip(xs, ys, strict=True):
        yi, xi = int(y), int(x)
        rgb = (
            (target[yi, xi] + 0.6 * lacking[yi, xi])
            * lift
            * study.rng.uniform(0.94, 1.06)
        )
        paint = study.paint(rgb)
        groups[id(paint)].append((x, y))
        paints[id(paint)] = paint
    batches = []
    for key, pts in groups.items():
        batches.extend((paints[key], pts[k : k + 4]) for k in range(0, len(pts), 4))
    sponge = brushes.SPONGE.sized(size)
    for k in study.rng.permutation(len(batches)):
        paint, pts = batches[k]
        study.c.stipple(
            pts, paint, amount=3.0 * len(pts) + 4, pressure=pressure, brush=sponge
        )
    return len(xs)


def foam(study, where, threshold=5.0):
    """The brightest foam on the falling water: small strokes of the palette
    knife, following the flow."""
    target, lacking, err = study.gap(10, 1.5)
    xs, ys = study.points(6, where & (err > threshold))
    knife = brushes.KNIFE_SMALL
    for x, y in zip(xs, ys, strict=True):
        a, _ = study.ref.direction(x, y, scale=4)
        d = study.rng.uniform(4, 8)
        pts = [
            (x - d * np.cos(a), y - d * np.sin(a)),
            (x + d * np.cos(a), y + d * np.sin(a)),
        ]
        yi, xi = int(y), int(x)
        paint = study.paint(target[yi, xi] + 0.6 * lacking[yi, xi])
        study.c.stroke(pts, paint, amount=5, brush=knife, pressure=0.5, angle=90)
    return len(xs)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    # Courbet worked up from a dark brown ground
    c = Canvas(W, h, surface=surfaces.LINEN_FINE.with_(ground="#2e2620"), seed=1864)
    study = Study(c, ref, seed=1864, body=0.55)
    ys, xs = np.mgrid[0:h, 0:W]
    value = np.asarray(ref.value(xs, ys, scale=3))
    water_mask = (value > 0.6) & (ys > 0.4 * h) & (xs > 0.2 * W)
    log("lay-in")
    study.layer(56, brushes.FLAT_HOG, straight=True, load=0.5, pressure=0.5)
    c.dry()
    # everything sponged, big sponges to small ones: pressed lightly at first,
    # so each print is a lacy web and they build up a grain (the dark showing
    # through), then the smallest sponge pressed harder for the edges; each
    # pass let dry so the next does not lift it again
    n = 0
    passes = (
        (44, 12, 6, 0.0, 0.2),
        (44, 12, 6, 3.0, 0.2),
        (30, 8, 4, 3.0, 0.2),
        (20, 6, 2.5, 4.0, 0.22),
        (14, 4.5, 1.5, 5.0, 0.25),
        (10, 3.5, 1.0, 6.0, 0.3),
    )
    everywhere = np.ones((h, W), bool)
    for size, spacing, scale, threshold, pressure in passes:
        log(f"sponge {size}px ({n} prints so far)")
        n += sponged(
            study, size, spacing, scale, everywhere, pressure, threshold=threshold
        )
        c.dry()
    log("knife: the foam")
    k = foam(study, water_mask & (value > 0.75))
    c.dry()
    # a brush only for the darkest thin lines
    log("the cracks, the timbers, the windows")
    study.layer(3, brushes.LINER, threshold=12.0, length=8, load=0.6, pressure=0.45,
                where=value < 0.3)  # fmt: skip
    c.save(path, relief=0.45, gloss=0.5)
    print(f"wrote {path}: {study.strokes} brush strokes, {n} sponge prints, "
          f"{k} knife strokes, in {time.time() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "courbet_loue.png")
    )
